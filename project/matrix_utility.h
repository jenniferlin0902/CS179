//
// Created by Jennifer Lin on 5/22/16.
//

#ifndef MATRIX_FACTORIZE_MATRIX_UTILITY_H
#define MATRIX_FACTORIZE_MATRIX_UTILITY_H

#include "common.h"

class matrix_utility {
public:
    static void convolve2D(float* f, float* input, float* result, dim_t x_dim, dim_t f_dim);
    static void convolve(float* a, float*b, float* result, int size);
    static void division(float* numerator, float* denumerator, float* result, int size);
    static float det(float* mat, int size);
};


#endif //MATRIX_FACTORIZE_MATRIX_UTILITY_H
