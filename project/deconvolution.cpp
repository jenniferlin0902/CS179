//
// Created by Jennifer Lin on 5/22/16.
//
#include "matrix_utility.h"
#include "cuda_matrix_utility.h"
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <cuda_runtime.h>

const int FIR_P = 3;



void FIR_normalization(float* FIR, int size){
    double sum = 0;
    for (int i = 0; i < size; i++){
        sum+=FIR[i];
    }
    for (int i =0; i < size; i++){
        FIR[i] = FIR[i] / (int)sum;
    }
}

void init_Weiner(float* FIR, int size){
    for (int i = 0; i < size; i++){
        FIR[i] = (float)(rand()%(256));
    }
    FIR_normalization(FIR, size);
}


void weiner_Rxy(float* x, float* y, float* Rxy, dim_t x_dim, dim_t f_dim){
    int size = x_dim.w * x_dim.h;
    for(int n = 0; n < f_dim.w * f_dim.h; n++){
        Rxy[n] = 0.0;
        int k1 = n % f_dim.w - (f_dim.w - 1)/2;
        int k2 = n % f_dim.h - (f_dim.h - 1)/2;
        for(int w = 0; w < x_dim.w - abs(k1); w++){
            for(int h = 0; h < x_dim.h - abs(k2); h++){
                float y_element = ((w - k1) < 0 || (h - k2) < 0) ? 0 : y[(w - k1) + (h - k2) * x_dim.w];
                Rxy[n] += x[h*x_dim.w + w] * y_element/(float)(256*size);
            }
        }
    }
}


void weiner_update_cpu(float* FIR, float* est, float* y, float* Ryy, dim_t x_dim, dim_t FIR_dim){
    int size = FIR_dim.w * FIR_dim.h;
    int P = (FIR_dim.w - 1)/2;
    float* Rxy = new float[size];
    weiner_Rxy(est, y, Rxy, x_dim, FIR_dim);

    for(int i = 0; i < size; i++){
        FIR[i] = 0;
        int r1 = i % FIR_dim.w;
        int r2 = i / FIR_dim.w;
        for(int n = 0; n < size; n++) {
            int t1 = n % FIR_dim.w;
            int t2 = n / FIR_dim.w;

            int x = (P + t1 - r1);
            int y = (P + t2 - r2);
            float Ryy_element = (x > 0 && y > 0) ? Ryy[y * FIR_dim.w + x] : 0.0;
            FIR[i] += Rxy[n] * Ryy_element;
        }
    }
    FIR_normalization(FIR, size);
    delete Rxy;
}


float MMSE_estimation(float est, float noise_var){
    float estimator;
    const float a = 255;
    const float p0 = 0.5;
    // logistic function between 0 to 255
    estimator = a / (1 + expf(-0.04*(est - 255/2)));
    return estimator;
}

void run_deconvolution_gpu(float* input, float* output, dim_t input_dim, int max_steps){
    dim_t FIR_dim = {2*FIR_P+1,2*FIR_P+1 };
    float* weiner_FIR = new float[FIR_dim.w * FIR_dim.h];
    float* dev_f;
    float* dev_input;
    float* dev_result;
    float* dev_Ryy;

    int size_x = input_dim.w * input_dim.h;
    int size_f = FIR_dim.w * FIR_dim.h;
    // initailize fileter in CPU side
    init_Weiner(weiner_FIR, (FIR_P*2+1)*(FIR_P*2+1));

    // allocate and initialize dev variables
    cudaMalloc(&dev_f, sizeof(float) * size_f);
    cudaMalloc(&dev_input, sizeof(float) * size_x);
    cudaMalloc(&dev_result, sizeof(float) * size_x);
    cudaMalloc(&dev_Ryy, sizeof(float)*size_f);

    cudaMemset(dev_Ryy, 0x0, sizeof(float)* size_f);
    cudaMemcpy(dev_f, weiner_FIR, sizeof(float) * size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_input, input, sizeof(float) * size_x, cudaMemcpyHostToDevice);
    cudaMemset(dev_result, 0x0, sizeof(float) * size_x);

    cuda_matrix_utility::weiner_Rxy(dev_input, dev_input,dev_Ryy, input_dim, FIR_dim);

    for(int i = 0; i < max_steps; i++){
        // apply linear filter
        cuda_matrix_utility::convolve2D(dev_f, dev_input, dev_result, input_dim, FIR_dim);

        // estimate filter result
        cuda_matrix_utility::MMSE_estimation(dev_result, input_dim.w * input_dim.h);
        // updat fileter
        cuda_matrix_utility::weiner_update(dev_f,dev_result, dev_input, dev_Ryy, input_dim, FIR_dim);
    }

    cudaMemcpy(output, dev_result, sizeof(float)*size_x, cudaMemcpyDeviceToHost);
    cudaFree(dev_result);
    cudaFree(dev_input);
    cudaFree(dev_f);
    cudaFree(dev_Ryy);

    std::cout<<"done GPU deconvolution" << std::endl;
    delete weiner_FIR;
}


void run_deconvolution_cpu(float* input, float* output, dim_t input_dim, int max_steps){
    dim_t FIR_dim = {2*FIR_P+1,2*FIR_P+1 };
    float* weiner_FIR = new float[FIR_dim.w * FIR_dim.h];
    float* Ryy = new float[FIR_dim.w * FIR_dim.h];

    // initialize weiner filter with random number
    init_Weiner(weiner_FIR, (FIR_P*2+1)*(FIR_P*2+1));
    // calculate cross correlation for input
    weiner_Rxy(input, input,Ryy, input_dim, FIR_dim);

    // start blind deconvolution step
    for(int i = 0; i < max_steps; i++){
        // apply filter
        matrix_utility::convolve2D(weiner_FIR, input, output, input_dim, FIR_dim);

        // estimate the filtered image
        for (int n = 0; n < input_dim.w*input_dim.h; n++){
            output[n] = MMSE_estimation(output[n],0.5);
        }
        // update filter value based on the estimation.
        weiner_update_cpu(weiner_FIR,output, input, Ryy, input_dim, FIR_dim);
    }
    std::cout<<"done CPU deconvolution" << std::endl;
    delete weiner_FIR;
}

