//
// Created by Jennifer Lin on 5/29/16.
//

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include "common.h"
#include "cuda_matrix_utility.cuh"
#include "cuda_matrix_utility.h"



void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){

        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

/**
 * 2D convolution between the image and weiner filter.
 */
void cuda_matrix_utility::convolve2D(float* dev_f, float* dev_input, float* dev_result, dim_t x_dim, dim_t f_dim){
    call2DConvolveKernal(dev_f, dev_result, dev_input, x_dim.w, x_dim.h, f_dim.w, f_dim.h);

}

/**
 * Nonlinear estimation for binary text image. We use a logistic function
 * to approximate.
 */
void cuda_matrix_utility::MMSE_estimation(float* data, float size){
    callMMSEEstKernal(data, size);
}

/**
 * Calculate cross correlation between dev_x and dev_y, and store in dev_xy
 */
void cuda_matrix_utility::weiner_Rxy(float* dev_x, float* dev_y, float* dev_xy, dim_t x_dim, dim_t f_dim){
    int f_size = f_dim.w * f_dim.h;
    for (int n = 0; n < f_size; n++){
        int k1 = n % f_dim.w - (f_dim.w - 1)/2;
        int k2 = n % f_dim.h - (f_dim.h - 1)/2;
        callWeinerRxyKernal(dev_x, dev_y, &(dev_xy[n]), x_dim.w, x_dim.h, k1, k2);
    }

}

/*
 * Update the weiner FIR filter based on the non linear estimation (dev_est)
 * in order to minimize the mean square error
 */
void cuda_matrix_utility::weiner_update(float*dev_f, float* dev_est, float* dev_input
                                          ,float* dev_Ryy, dim_t x_dim, dim_t f_dim){
    float* dev_Rxy;
    double* dev_sum;
    int f_size = f_dim.w * f_dim.h;
    cudaMalloc(&dev_Rxy, sizeof(float)*f_size);
    cudaMemset(dev_Rxy, 0x0, sizeof(float)*f_size);

    cuda_matrix_utility::weiner_Rxy(dev_est, dev_input, dev_Rxy, x_dim, f_dim);
    callWeinerUpdateKernal(dev_f, dev_Rxy, dev_Ryy, f_dim.w, f_dim.h);
    callFIRNormalizeKernal(dev_f, f_size);
    cudaFree(dev_Rxy);

}