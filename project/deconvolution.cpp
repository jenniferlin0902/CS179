//
// Created by Jennifer Lin on 5/22/16.
//
#include "matrix_utility.h"
#include "cuda_matrix_utility.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>

const int FIR_P = 3;



void FIR_normalization(float* FIR, int size){
    int sum = 0;
    for (int i = 0; i < size; i++){
        sum+=FIR[i];
    }
    for (int i =0; i < size; i++){
        FIR[i] = FIR[i] / (float)sum;
    }
}

void init_Weiner(float* FIR, int size){
    for (int i = 0; i < size; i++){
        FIR[i] = (float)(rand()%(256));
    }
    FIR_normalization(FIR, size);
}


void weiner_Rxy(float* x, float* y, float* Rxy, dim_t x_dim, dim_t f_dim){
    //float* Ry = new float[size];
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

void weiner_update(float* FIR, float* est, float* y, float* Ryy, dim_t x_dim, dim_t FIR_dim){
    int size = FIR_dim.w * FIR_dim.h;
    int P = (FIR_dim.w - 1)/2;
    float* Rxy = new float[size];
    //float* Ryy = new float[1];
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
    estimator = a / (1 + expf(-0.04*(est - 255/2)));
/*
    if (est < 255/2){
        estimator = 1;
    //}else if (est > 200){
      //  estimator = 255;
    }else{
        estimator = 255;
    }*/

    return estimator;
}

void run_deconvolution(float* input, float* output, dim_t input_dim, int max_steps){
    dim_t FIR_dim = {2*FIR_P+1,2*FIR_P+1 };
    float* weiner_FIR = new float[FIR_dim.w * FIR_dim.h];
    float* Ryy = new float[FIR_dim.w * FIR_dim.h];
    init_Weiner(weiner_FIR, (FIR_P*2+1)*(FIR_P*2+1));
    weiner_Rxy(input, input,Ryy, input_dim, FIR_dim);
    for(int i = 0; i < max_steps; i++){
        // linear estimation step
        cuda_matrix_utility::convolve2D(weiner_FIR, input, output, input_dim, FIR_dim);
        //nolinear estimatin step
        //matrix_utility::division(output, input, output, w*h);
        //float max = 0;
        for (int n = 0; n < input_dim.w*input_dim.h; n++){
            output[n] = MMSE_estimation(output[n],0.5);
        }
        weiner_update(weiner_FIR,output, input, Ryy, input_dim, FIR_dim);
    }
    std::cout<<"done deconvolution" << std::endl;
    delete weiner_FIR;
}

