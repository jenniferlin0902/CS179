#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"
#include <cstdio>


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16
 * blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */


/* Reading from input is coalesced, but writing to output, each thread
 * is accessing from different row at the same time. Thus, the
 * write is not coalesced. Each loop will touch 32 cache line.
 * */

__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;
    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];


}




/* First, read in to shared memory. All read are coaleaced, since each thread
 * are accessing from the same row */
/* Note that input is stored in shared memory data[] in a shifted manner.
 * That is, data [0,0] in bank 0, data [0,1] in bank 1, data[i,j] in bank (j%32).
 * This is for avoiding bank conflict when writing to output. During each
 * loop, the thread in warp will write to the subsequent bank */
/*
 * Read each thread read from shared memory. For each thread, it will write
 * to a column to output[], so global memory access is coalesced.
 * Note that each thread should read from a row in
 * shared memory. Since the data is stored shifted. For example, data [0,0]
 * is bank 0 and data [0,1] is in bank 1, so there is no bank conflict; thread
 * 0 and thread 1 is accessing different bank.
 */

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {

    __shared__ float data[64*64*2];

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 *threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    int y = threadIdx.y * 4;
    int x = threadIdx.x + y;
    for (; j < end_j; j++) {
        data[ x + y * ( 64 * 2)] = input[i + n * j];
        y++;
        x++;
    }

    __syncthreads();

    y = threadIdx.x;
    x = threadIdx.y * 4 + y;
    int i1 = threadIdx.x + 64 * blockIdx.y;
    int j1 = 4 *threadIdx.y + 64 * blockIdx.x;
    int end_j1 = j1 + 4;
    for (; j1 < end_j1; j1++){
        output[i1 + n * j1] = data[ x + y * (64 * 2)];
        x++;
    }
}


/* The optimalTransposeKernel is build on shmemTranspose, so all
 * global memory is coalesced and no bank conflict. It applies
 * loop unwinding, ILP, and minimize duplicate arithmatic.
 * */
__global__
 
void optimalTransposeKernel(const float *input, float *output, int n) {
    __shared__ float data[64*64*2];

    const int i = threadIdx.x + (blockIdx.x << 6);
    int j = (threadIdx.y << 2) + (blockIdx.y << 6);
    int base = threadIdx.x + threadIdx.y * 516;

    data[base] = input[i + n*(j)];
    data[base + 129] = input[i + n*(j+1)];
    data[base + 258] = input[i + n*(j+2)];
    data[base + 387] = input[i + n*(j+3)];

    __syncthreads();

    base = (threadIdx.x << 7) + (threadIdx.y << 2) + threadIdx.x;
    int i1 = threadIdx.x + (blockIdx.y << 6);
    int j1 = (threadIdx.y << 2) + (blockIdx.x << 6);

    output[i1 + n * (j1)] = data[base];
    output[i1 + n * (j1+1)] = data[base + 1];
    output[i1 + n * (j1+2)] = data[base + 2];
    output[i1 + n * (j1+3)] = data[base + 3];

}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
