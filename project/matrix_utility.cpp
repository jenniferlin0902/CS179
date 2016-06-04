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
                int x_element = (x_index >= 0 && y_index >= 0) ? input[x_index + y_index * x_dim.w] : 0;
                result[y * x_dim.w + x] += f[n] * x_element;
                if (result[y*x_dim.w + x] > 255){
                    break;
                }
            }
        }
    }
}

