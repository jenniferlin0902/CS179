//
// Created by Jennifer Lin on 5/29/16.
//


#include "cuda_matrix_utility.cuh"

void cuda_matrix_utility::convolve2D(float* f, float* input, float* result, dim_t x_dim, dim_t f_dim){
    float* dev_f;
    float* dev_input;
    float* dev_result;
    int size_x = x_dim.w * x_dim.h;
    int size_f = f_dim.w * f_dim.h;
    cudaMalloc(&dev_f, sizeof(float) * size_f);
    cudaMalloc(&dev_input, sizeof(float) * size_x);
    cudaMalloc(&dev_result, sizeof(float) * size_x);

    cudaMemcpy(dev_f, f, sizeof(float) * size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_input, input, sizeof(float) * size_x, cudaMemcpyHostToDevice);
    cudaMemset(dev_result, 0x0, sizeof(float) * size_x);

    call2DConvolveKernal(dev_f, dev_result, dev_input, x_dim.w, x_dim.h, f_dim.w, f_dim.h);

    cudaMemcpy(result, dev_result, sizeof(float)*size_x, cudaMemcpyDeviceToHost);

    cudaFree(dev_f);
    cudaFree(dev_input);
    cudaFree(dev_result);
}