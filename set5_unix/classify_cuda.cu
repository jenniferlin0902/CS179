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

    __shared__ float grad[REVIEW_DIM];
    //float* grad = shared;
    float temp;
    float* err_count = &temp;
    // float* err_count = (float*)(grad + REVIEW_DIM);
    int x_dim = REVIEW_DIM + 1;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;


    if (threadId < REVIEW_DIM){
        grad[threadId] = 0.0;
        //printf("data = %f\n", data[threadId*x_dim + 50]);
    }
    if (threadId == 0){
        *err_count = 0;
        printf("err_cout = %f\n", *err_count);
    }
    __syncthreads();
    while (threadId < batch_size){
        float y = data[threadId * x_dim + 50];

        for (int i = 0; i < REVIEW_DIM; i++){
            float x = data[threadId * x_dim + i];
            //printf("x = %f\n", x);
            float sub_grad =  x * y / (1 + exp(y * weights[threadId] * x));
            atomicAdd(&(grad[i]), sub_grad);

        }
        threadId += blockDim.x * gridDim.x;
        printf("grad = %f\n", grad[1]);
    }
    __syncthreads();
    threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId == 0){
        for (int i = 0; i < REVIEW_DIM; i++){
           // weights[i] -= step_size * grad[i]/batch_size;
        }
    }

/*
    while (threadId < batch_size){
        float est_y = 0;
        for (int i = 0; i < REVIEW_DIM; i++){
            est_y +=  data[threadId * x_dim + i]* weights[i];
        }
        printf("est_y = %f\n", est_y);
        if (est_y * data[threadId * x_dim + 50] <= 0 ){
            atomicAdd(err_count, 1);
            //printf("here\n");
        }
        threadId += blockDim.x * gridDim.x;
    }
*/
    __syncthreads();
    threadId = blockIdx.x * blockDim.x + threadIdx.x;
    //if (threadId == 0){
        //int temp = *err_count/batch_size;
        *errors = 1.0;

}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(
    float *data,
    int batch_size, 
    float step_size,
    float *weights, 
    cudaStream_t stream)
{
    int block_size = (batch_size < 1024) ? batch_size : 1024;

    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;
    int shmem_bytes = 0;



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
