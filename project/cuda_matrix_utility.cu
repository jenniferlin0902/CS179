
#include <cuda_runtime.h>
#include <cufft.h>
#include "common.h"
#include "cuda_matrix_utility.cuh"


__global__
void cuda2DConvolveKernal(float* f, float* result, float* input,
                          int input_x, int input_y, int f_x, int f_y){
    uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int Px = (f_x - 1)/2;
    int Py = (f_y - 1)/2;
    while(thread_id < input_x * input_y){
        // TODO load f into shared memory?
        int x = thread_id % input_x;
        int y = thread_id / input_x;
        for(int n = 0; n < f_x * f_y; n++){
            int t1 = n % f_dim.w;
            int t2 = n / f_dim.w;
            int x_index = x - t1 + Px;
            int y_index = y - t2 + Py;
            float x_element = (x_index >= 0 && y_index >= 0) ? input[x_index + y_index * x_dim.w] : 0;
            result[y * x_dim.w + x] += f[n] * x_element;
        }
        thread_id += blockDim.x * gridDim.x;
    }
}


void call2DConvolveKernal(float* f, float* result, float* input,
                          int input_x, int input_y, int f_x, int f_y){
    // max threads per block is 1024, nblocks = 512
    int input_size = input_x * input_y;
    int block_size = input_size < 1024 ? input_size : 1024;
    int nblocks = input_size / block_size < 512 ? input_size / block_size : 512;
    cuda2DConvolveKernal<<<block_size, nblocks>>>(f, result, input, input_x, input_y, f_x, f_y);
}
