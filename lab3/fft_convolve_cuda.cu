/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/*
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_index < padded_length) {
        cufftComplex impulse = impulse_v[thread_index];
        cufftComplex raw = raw_data[thread_index];
        cufftComplex out;
        out.x = (impulse.x * raw.x - impulse.y * raw.y)/padded_length;
        out.y = (impulse.y * raw.x + impulse.x * raw.y)/padded_length;
        out_data[thread_index] = out;
        thread_index += blockDim.x * gridDim.x;
    }
}


__device__ float complexCompare(cufftComplex num1, cufftComplex num2){
    float mag1 = abs(num1.x);
    float mag2 = abs(num2.x);
    return (mag1 >= mag2) ? mag1 : mag2;
}


/**
 * cudaMaximumKernel optimize maximum finding with reduction
 *
 * Each block handle block * 2 size of out_data. The reduction
 * is performed in shared data s_max_data, on each block of data.
 * First, load the max of magnitude of out_data[i] and out_data[i + blockDim.x]
 * to the shared memory. Each thread read from out_data consecutavely. Thus, gloval
 * memory is coalesced. Each thread store the 4 bybte data in s_max_data[threadId].
 * Thus, within a warp, each thread, assigned a different bank. There is no
 * bank conflict.
 * Next, use a for loop to reduce the size of share memory.
 * In each, each thread access the shared memory sequentially. Thus,
 * thre is no bank conflict. In addition, note that the thread is used
 * sequentially as well.
 * When the reduction reach below 32 threads, I unroll the for loop. Since
 * all threads in a warp is syncronize, we don't need to sync threads between
 * each step.
 * The reduction will find the local maximum within a block. Every block will
 * perform atomic compare and swap on the global variable max_abs_val to get the
 * global max.
 */

__global__
void cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    extern __shared__ float s_max_data[];

    uint threadId = threadIdx.x;
    uint i = blockIdx.x * (blockDim.x * 2)+ threadIdx.x;
    while((i + blockDim.x) < padded_length) {
        s_max_data[threadId] = complexCompare(out_data[i], out_data[i + blockDim.x]);
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadId < s) {
                s_max_data[threadId] = (s_max_data[threadId] >= s_max_data[threadId + s])\
 ? s_max_data[threadId] : s_max_data[threadId + s];
                __syncthreads();
            }
        }
        if (threadId == 0) {
            atomicMax(max_abs_val, s_max_data[0]);
        }
    }
}

/**
 * cudaDivideKernal load the outdata to local register first,
 * perform all the necessary calculation and write it back into
 * global memory. Note that all memory access is sequential so glbal
 * memory read/write is coalesced.
 */
__global__
void cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float max = *max_abs_val;
    while (thread_index < padded_length) {
        cufftComplex data = out_data[thread_index];
        data.x = data.x/max;
        data.y = data.y/max;
        out_data[thread_index] = data;
        thread_index += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

        cudaMaximumKernel<<<blocks, threadsPerBlock, sizeof(float)*threadsPerBlock>>>(out_data, max_abs_val, padded_length);


}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
    cudaDivideKernel <<< blocks, threadsPerBlock >>> (out_data, max_abs_val, padded_length);

}