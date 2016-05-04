#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"
#include "math.h"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */


__global__
void trainLogRegKernel(
    float *data,
    int batch_size,
    int step_size,
	float *weights,
    float *errors)
{

    extern __shared__ float grad[];

    int x_dim = REVIEW_DIM + 1;
    int threadId = threadIdx.x;

    while (threadId < REVIEW_DIM){
        grad[threadId] = 0.0;
        threadId += blockDim.x;
       // printf("loop1\n");
    }
   // printf("out of loop\n");
    __syncthreads();

    // Assign a data to each thread. Need to handle batch_size of data

    threadId = blockIdx.x * blockDim.x + threadIdx.x;
    while (threadId < batch_size){
        float y = data[threadId * x_dim + 50];
        // calculate dot product
        float dot = 0;

        for (int i = 0; i < REVIEW_DIM; i++){
            dot += weights[i]*data[threadId * x_dim + i];
        }
        for (int i = 0; i < REVIEW_DIM; i++){
            float x = data[threadId * x_dim + i];
            //printf("x = %f\n", x);
            float sub_grad =  x * y / (1.0 + exp(y * dot));
            atomicAdd(&(grad[i]), sub_grad);
        }
        threadId += blockDim.x * gridDim.x;
        //printf("loop2\n");
    }
    __syncthreads();

    // update weight
    threadId = threadIdx.x;
    while (threadId < REVIEW_DIM){
        //printf("grad[%d] = %f\n", threadId, grad[threadId]/(float)batch_size*step_size);
        atomicAdd(&(weights[threadId]), (grad[threadId]/(float)batch_size)*step_size);
        threadId += blockDim.x;
    }
    __syncthreads();

    // get estimate y and calculate error
    threadId = blockIdx.x * blockDim.x + threadIdx.x;
    while (threadId < batch_size){
        float est_y = 0;
        float y = data[threadId * x_dim + 50];
        for (int i = 0; i < REVIEW_DIM; i++){
            est_y +=  data[threadId * x_dim + i]* weights[i];
        }
        if (est_y * y <= 0 ){
            atomicAdd(errors, 1.0);
        }
        threadId += blockDim.x * gridDim.x;
        //printf("loop3\n");
    }
    __syncthreads();

    threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId == 0){
       // printf("before error %f\n", *errors);
        *errors = *errors/batch_size;
        printf("final error %f\n", *errors);
    }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(
    float *data,
    int batch_size, 
    int step_size,
    float *weights,
    cudaStream_t stream)
{
    int block_size = (batch_size < 1024) ? batch_size : 1024;

    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;
    int shmem_bytes = (REVIEW_DIM) * sizeof(float);


    float *d_errors;
    cudaMalloc(&d_errors, sizeof(float));
    cudaMemset(d_errors, 0, sizeof(float));

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);
    float h_errors = -1.0;
    cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
    cudaFree(d_errors);
    return h_errors;
}
