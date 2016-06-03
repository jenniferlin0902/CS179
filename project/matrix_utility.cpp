//
// Created by Jennifer Lin on 5/22/16.
//

#include "matrix_utility.h"
#include <algorithm>

// This assume the same size for x and y
void matrix_utility::convolve(float* x, float* y , float* result, int size){
    for(int i = 0; i < size; i++){
        result[i] = 0;
        for(int n = 0; n < size; n++){
            result[i] += x[n] * y[(i - n + size) % size];
        }
    }
}

void matrix_utility::convolve2D(float* f, float* input, float *result, dim_t x_dim, dim_t f_dim) {
    int Py = (f_dim.h - 1)/2;
    int Px = (f_dim.w - 1)/2;
    for (int x = 0; x < x_dim.w; x++){
        for(int y = 0; y < x_dim.h; y++){
            result[y * x_dim.w + x] = 0;
            for(int n = 0; n < f_dim.w * f_dim.h; n++){
                int t1 = n % f_dim.w;
                int t2 = n / f_dim.w;
                int x_index = x - t1 + Px;
                int y_index = y - t2 + Py;
                float x_element = (x_index >= 0 && y_index >= 0) ? input[x_index + y_index * x_dim.w] : 0;
                result[y * x_dim.w + x] += f[n] * x_element;
            }
        }
    }
}

float matrix_utility::det(float* x, int size){
    float det = 0;
    float temp = 1.0;
    //std::fill(x, x+size, 0.0);
    //std::fill(temp, temp+size, 1.0);
    for (int i = 0; i < size - 1; i++) {
        for (int n = 0; n < size; n++) {
            temp *= x[n * size + (n + i) % size];
        }
        det += temp;
        temp = 1.0;
    }

    for (int i = 1; i < size; i++) {
        for (int n = 0; n < size; n++){
            temp *= x[n * size + (2*size - n -i) % size];
        }
        det -= temp;
        temp = 1.0;
    }
    return det;
}

void matrix_utility::division(float* numerator, float* denumerator, float* result, int size){
    for (int i = 0; i < size; i++){
        result[i] = numerator[i]/denumerator[i];
    }
}
