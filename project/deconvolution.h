//
// Created by Jennifer Lin on 5/22/16.
//

#ifndef MATRIX_FACTORIZE_DECONVOLUTION_H
#define MATRIX_FACTORIZE_DECONVOLUTION_H

#include "common.h"

void  run_deconvolution_cpu(float* input, float* output, dim_t input_dim, int steps);
void  run_deconvolution_gpu(float* input, float* output, dim_t input_dim, int steps);

#endif //MATRIX_FACTORIZE_DECOVOLUTION_H
