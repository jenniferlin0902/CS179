//
// Created by Jennifer Lin on 5/29/16.
//

#ifndef MATRIX_FACTORIZE_CUDA_MATRIX_UTILITY_H
#define MATRIX_FACTORIZE_CUDA_MATRIX_UTILITY_H
#include "common.h"

class cuda_matrix_utility {
public:
    static void convolve2D(float* f, float* input, float* result, dim_t x_dim, dim_t f_dim);
    static void MMSE_estimation(float* data, float size);
    static void weiner_Rxy(float* x, float* y, float* Rxy, dim_t x_dim, dim_t f_dim);
    static void weiner_update(float*dev_f, float* dev_est, float* dev_input
            ,float* dev_Ryy, dim_t x_dim, dim_t f_dim);

};


#endif //MATRIX_FACTORIZE_CUDA_MATRIX_UTILITY_H
