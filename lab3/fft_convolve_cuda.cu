/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>
#include <math.h>
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

__device__ static float atomicComplexMax(cufftComplex* address, cufftComplex val)
{
    float mag1 = (address->x)*(address->x) + (address->y)*(address->y);
    float mag2 = (val.x)*(val.x) + (val.y)*(val.y);
    return atomicMax(&mag1, mag2);
}


__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */

    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_index < padded_length) {
        cufftComplex impulse = impulse_v[thread_index];
        cufftComplex raw = raw_data[thread_index];
        out_data[thread_index].x = (impulse.x * raw.x - impulse.y * raw.y)/padded_length;
        out_data[thread_index].y = (impulse.y * raw.x + impulse.x * raw.y)/padded_length;
        thread_index += blockDim.x * gridDim.x;
    }
}

__device__ void warpMax(volatile float* sdata, int tid){
    sdata[tid] = atomicMax((float*)&sdata[tid], sdata[tid + 32]);
    sdata[tid] = atomicMax((float*)&sdata[tid], sdata[tid + 16]);
    sdata[tid] = atomicMax((float*)&sdata[tid], sdata[tid + 8]);
    sdata[tid] = atomicMax((float*)&sdata[tid], sdata[tid + 4]);
    sdata[tid] = atomicMax((float*)&sdata[tid], sdata[tid + 2]);
    sdata[tid] = atomicMax((float*)&sdata[tid], sdata[tid + 1]);
}

__global__
void cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)
    */
    extern __shared__ float s_max_data[];
    //extern __shared__ cufftComplex max_data[];
    uint threadId = threadIdx.x;

    while(threadId < padded_length){
        uint i = blockIdx.x * (blockDim.x * 2)+ threadIdx.x;
        s_max_data[threadId] = atomicComplexMax(&out_data[i],out_data[i + blockDim.x]);
        __syncthreads();

        for(unsigned int s = blockDim.x/2; s > 32; s >>= 1){
            if (threadId < s){
                s_max_data[threadId] = atomicMax((float *)s_max_data + threadId, \
                s_max_data[threadId + s]);
            }
        }
        if (threadId < 32){
            warpMax(s_max_data, threadId);
        }
        threadId += blockDim.x * gridDim.x;
    }

    if (threadId == 0){
        *max_abs_val = sqrt(s_max_data[0]);
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_index < padded_length) {
        float max = *max_abs_val;
        cufftComplex data = out_data[thread_index];
        out_data[thread_index].x = data.x/max;
        out_data[thread_index].y = data.y/max;
        thread_index += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        cudaProdScaleKernel<<<blocks, threadsPerBlock, padded_length * sizeof(cufftComplex)>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        cudaMaximumKernel<<<blocks, threadsPerBlock, padded_length * sizeof(float)>>>(out_data, max_abs_val, padded_length);
    /* TODO 2: Call the max-finding kernel. */

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        cudaDivideKernel<<<blocks, threadsPerBlock, padded_length>>>(out_data, max_abs_val, padded_length);
    /* TODO 2: Call the division kernel. */
}
