
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include "common.h"
#include "cuda_matrix_utility.cuh"


__global__
void cuda2DConvolveKernal(float* f, float* result, float* input,
                          int input_x, int input_y, int f_x, int f_y){
    uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int Px = (f_x - 1)/2;
    int Py = (f_y - 1)/2;
    extern __shared__ float shared_f[];

    if(thread_id < f_x * f_y){
      shared_f[thread_id] = f[thread_id];
    }

    while(thread_id < input_x * input_y){
        // TODO load f into shared memory?
        int x = thread_id % input_x;
        int y = thread_id / input_x;
        result[thread_id] = 0;
        for(int n = 0; n < f_x * f_y; n++){
            int t1 = n % f_x;
            int t2 = n / f_x;
            int x_index = x - t1 + Px;
            int y_index = y - t2 + Py;
            float x_element = (x_index >= 0 && y_index >= 0) ? input[x_index + y_index * input_x] : 0;
            result[thread_id] += f[n] * x_element;
        }
        thread_id += blockDim.x * gridDim.x;
    }
}


__global__
void cudaMMSEEstKernal(float* dev_data, int size){
    uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    while(thread_id < size){
        dev_data[thread_id] =  255/ (1 + expf(-0.04*(dev_data[thread_id] - 255/2)));
        thread_id += blockDim.x * gridDim.x;
    }
}

__global__
void cudaWeinerRxyKernal(float* x, float* y, float* Rxy,
                         int x_w, int x_h, int k1, int k2){
    uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = x_w * x_h;
    while(thread_id < size){
        // TODO load to shared memory ?
        int w = thread_id % x_w;
        int h = thread_id % x_h;
        if ((w < x_w - abs(k1)) && (h < x_h - abs(k2))){
            float y_element = ((w - k1) < 0 || (h - k2) < 0) ? 0 : y[(w - k1) + (h - k2) * x_w];
            atomicAdd(Rxy, x[h*x_w + w] * y_element/(float)(256*size));
        }
        thread_id += blockDim.x * gridDim.x;
    }
}

__global__
void cudaWeinerUpdateKernal(float* f, float* Rxy, float* Ryy, int f_w, int f_h){
    uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = f_w * f_h;
    int P = (f_w - 1)/2;
    while(thread_id < size){
        int r1 = thread_id % f_w;
        int r2 = thread_id / f_w;
        for(int n = 0; n < size; n++) {
            int t1 = n % f_w;
            int t2 = n / f_w;
            int x = (P + t1 - r1);
            int y = (P + t2 - r2);
            float Ryy_element = (x > 0 && y > 0) ? Ryy[y * f_w + x] : 0.0;
            f[thread_id] += Rxy[n] * Ryy_element;
        }
        thread_id += blockDim.x * gridDim.x;
    }

}

__global__
void cudaFIRNormalizeKernal(float* f, int f_size){
    __shared__ double temp[1];
    unsigned int threadId = threadIdx.x;
    if(threadId < f_size){
        temp[0] += f[threadId];
    }
    __syncthreads();

    /*
    for (unsigned int s = blockDim.x/2 ; s > 0; s >> 1){
        if(threadId < s){
            temp[threadId] = temp[threadId] + temp[threadId + s];
            __syncthreads();
        }
    } */

    if (threadId < f_size){
        f[threadId] = f[threadId] / temp[0];
    }
}


void call2DConvolveKernal(float* f, float* result, float* input,
                          int input_x, int input_y, int f_x, int f_y){
    // max threads per block is 1024, nblocks = 512
    int shmem = f_x * f_y * sizeof(float);
    int input_size = input_x * input_y;
    int block_size = input_size < 1024 ? input_size : 1024;
    int nblocks = input_size / block_size < 512 ? input_size / block_size : 512;
    cuda2DConvolveKernal<<<nblocks, block_size, shmem>>>(f, result, input, input_x, input_y, f_x, f_y);
}

void callMMSEEstKernal(float* data, int size){
    int block_size = (size < 1024) ? size : 1024;
    int nblocks = size/block_size < 512 ? size/block_size : 512;
    cudaMMSEEstKernal<<<nblocks, block_size>>>(data, size);
}

void callWeinerRxyKernal(float* x, float* y, float* Rxy,
                         int x_w, int x_h, int k1, int k2){
    int size = x_w * x_h;
    int block_size = 32;
    int nblocks = 32;
    cudaWeinerRxyKernal<<<block_size, nblocks>>>(x, y, Rxy, x_w, x_h, k1, k2);
}

void callWeinerUpdateKernal(float* f, float* Rxy, float* Ryy, int f_w, int f_h){
    int size = f_w * f_h;
    int block_size = (size < 1024) ? size : 1024;
    int nblocks = size/block_size;
    cudaWeinerUpdateKernal<<<1, size>>>(f, Rxy, Ryy, f_w, f_h);
}

void callFIRNormalizeKernal(float* f, int size){
    cudaFIRNormalizeKernal<<<2, 64>>>(f, size);
}