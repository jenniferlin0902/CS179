//
// Created by Jennifer Lin on 6/2/16.
//

#ifndef MATRIX_FACTORIZE_CUDA_MISC_UTILITY_H
#define MATRIX_FACTORIZE_CUDA_MISC_UTILITY_H


class cuda_misc_utility {
public:
    static void float2Complex(float* f, cufftComplex* c, int size);
    static void complex2float(float* f, cufftComplex* c, int size);
};


#endif //MATRIX_FACTORIZE_CUDA_MISC_UTILITY_H
