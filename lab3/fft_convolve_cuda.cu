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


__device__ float complexCompare(cufftComplex num1, cufftComplex num2){
    float mag1 = abs(num1.x);
    float mag2 = abs(num2.x);
    return (mag1 >= mag2) ? mag1 : mag2;
}

/*
__device__ void warpMax(volatile float* sdata, int tid){


}*/




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

    __shared__ float s_max_data[1024];

    uint threadId = threadIdx.x;
    uint i = blockIdx.x * (blockDim.x * 2)+ threadIdx.x;
    //printf("padded_length = %d\n", padded_length);
    float current_max =abs(out_data[i].x);
    //float step = ;
    while(i < padded_length){
       // uint step = blockDim.x;
        float temp_max = abs(out_data[i].x);
        if ((i + blockDim.x) < padded_length){
                float temp2 = abs(out_data[i + blockDim.x].x);
                temp_max = (temp_max > temp2) ? temp_max:temp2;
        }
        current_max = (current_max > temp_max ) ? current_max : temp_max;
        i += (blockDim.x * 2) * gridDim.x;
    }
    s_max_data[threadId] = current_max;
        __syncthreads();
        //printf("first copy\n");
        for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
            if (threadId < s){
                s_max_data[threadId] = (s_max_data[threadId] >= s_max_data[threadId + s])\
                        ? s_max_data[threadId] : s_max_data[threadId + s];
            }
            __syncthreads();
        }
        if (threadId < 32){
            s_max_data[threadId] = (s_max_data[threadId] >= s_max_data[threadId + 32]) \
                    ? s_max_data[threadId] : s_max_data[threadId + 32];
            s_max_data[threadId] = (s_max_data[threadId] >= s_max_data[threadId + 16]) \
                    ? s_max_data[threadId] : s_max_data[threadId + 16];
            s_max_data[threadId] = (s_max_data[threadId] >= s_max_data[threadId + 8]) \
                    ? s_max_data[threadId] : s_max_data[threadId + 8];
            s_max_data[threadId] = (s_max_data[threadId] >= s_max_data[threadId + 4]) \
                    ? s_max_data[threadId] : s_max_data[threadId + 4];
            s_max_data[threadId] = (s_max_data[threadId] >= s_max_data[threadId + 2]) \
                    ? s_max_data[threadId] : s_max_data[threadId + 2];
            s_max_data[threadId] = (s_max_data[threadId] >= s_max_data[threadId + 1]) \
                    ? s_max_data[threadId] : s_max_data[threadId + 1];

        }
        if (threadId == 0){
            atomicMax(max_abs_val, s_max_data[0]);
        }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
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
        cudaMaximumKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
    /* TODO 2: Call the max-finding kernel. */

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
    /* TODO 2: Call the division kernel. */
}
