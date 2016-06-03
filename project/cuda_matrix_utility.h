//
// Created by Jennifer Lin on 5/29/16.
//

#ifndef MATRIX_FACTORIZE_CUDA_MATRIX_UTILITY_H
#define MATRIX_FACTORIZE_CUDA_MATRIX_UTILITY_H
#include "common.h"

class cuda_matrix_utility {
public:
    static void convolve2D(float* f, float* input, float* result, dim_t x_dim, dim_t f_dim);
};


#endif //MATRIX_FACTORIZE_CUDA_MATRIX_UTILITY_H
