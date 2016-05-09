/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


/* TODO: You'll need a kernel here, as well as any helper functions
to call it */
__global__
void cuda1DFDWaveKernal(
        float* dev_new,
        float* dev_current,
        float* dev_old,
        float courandSquared,
        uint numberOfNodes
    ){
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    while (threadId < numberOfNodes - 3){
        uint index = threadId + 1;
        dev_new[index] = 2 * dev_current[index] - dev_old[index] + courandSquared * (
                dev_current[index + 1] - 2 * dev_current[index] + dev_current[ index - 1]);

        //dev_new[threadId] = 1.0;
        threadId += blockDim.x * gridDim.x;
    }
}

__global__
void cuda1DWaveBCKernal(
        float* dev_new,
        float left_bc,
        uint numberOfNodes){
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId == 0){
        dev_new[0] = left_bc;
        dev_new[numberOfNodes - 1] = 0;
    }
}


void callCuda1DFDWaveKernal(
        float* dev_new,
        float* dev_current,
        float* dev_old,
        float courantSquared,
        uint numberOfNodes,
        uint threadPerBlock,
        uint blocks){
    cuda1DFDWaveKernal<<<blocks, threadPerBlock>>>(dev_new, dev_current, dev_old,courantSquared, numberOfNodes);
}

void callCuda1DWaveBCKernal(
        float* dev_new,
        float left_bc,
        uint numberOfNodes,
        uint threadPerBlock,
        uint blocks){
    cuda1DWaveBCKernal<<<blocks, threadPerBlock>>>(dev_new,left_bc, numberOfNodes);
}
