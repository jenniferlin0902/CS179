#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"
#include <cstdio>
/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


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
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 2 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 2;
    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
    /* Reading from input is coalesced, but writing to output, each thread
     * is accessing from different row at the same time. Thus, the
     * write is not coalesced. Each loop will touch 128 cache line.
     * */

}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory


    __shared__ float data[64*64*2];

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 *threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;
    /* First, read in to shared memory. All read are coaleaced, since each thread
     * are accessing from the same row */
    /* Note that input is stored in shared memory data[] in a shifted manner.
     * This is for avoiding bank conflict when writing to output. During each
     * loop, the thread in warp will write to the subsequent bank */
    int y = threadIdx.y * 4;
    int x = threadIdx.x + y;
    for (; j < end_j; j++) {
        data[ x + y * ( 64 * 2)] = input[i + n * j];
        y++;
        x++;
    }

    __syncthreads();
    /*
     * Read each thread read from shared memory. For each thread, it will write
     * to a column to output[], so globla meory acess is coalesced.
     * Note that each thread should read from a row in
     * shared memory. Since the data is stored shifted. For example, data [0,0]
     * is bank 0 and data [0,1] is in bank 1, so there is no bank conflict; thread
     * 0 and thread 1 is accessing different bank.
     */
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

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {    __shared__ float data[64*64*2];
    /*
    const int i = threadIdx.x + (blockIdx.x << 6);
    int j = (threadIdx.y << 2) + (blockIdx.y << 6);
    float temp1, temp2, temp3, temp4;
    int base = threadIdx.x + threadIdx.y * 516;

    temp1 = input[i + n*(j)];
    temp2 = input[i + n*(j+1)];
    temp3 = input[i + n*(j+2)];
    temp4 = input[i + n*(j+3)];

    data[base] = temp1;
    data[base + 129] = temp2;
    data[base + 258] = temp3;
    data[base + 387] = temp4;

    __syncthreads();

    //y = threadIdx.x;
    //x = threadIdx.y * 4 + y;
    base = (threadIdx.x << 7) + (threadIdx.y << 2) + threadIdx.x;
    int i1 = threadIdx.x + (blockIdx.y << 6);
    int j1 = (threadIdx.y << 2) + (blockIdx.x << 6);

    temp1 = data[base];
    temp2 = data[base + 1];
    temp3 = data[base + 2];
    temp4 = data[base + 3];

    output[i1 + n * (j1)] = temp1;
    output[i1 + n * (j1+1)] = temp2;
    output[i1 + n * (j1+2)] = temp3;
    output[i1 + n * (j1+3)] = temp4;
*/

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = threadIdx.y*4 + 64 * blockIdx.y;
    const int end_j = j + 4;
    /* First, read in to shared memory. All read are coaleaced, since each thread
     * are accessing from the same row */
    /* Note that input is stored in shared memory data[] in a shifted manner.
     * This is for avoiding bank conflict when writing to output. During each
     * loop, the thread in warp will write to the subsequent bank */
    int y = threadIdx.y * 4;
    int x = threadIdx.x + y;
    for (; j < end_j; j++) {
        data[ x + y * ( 64 * 2)] = input[i + n * j];
        y++;
        x++;
    }

    __syncthreads();
    /*
     * Read each thread read from shared memory. For each thread, it will write
     * to a column to output[], so globla meory acess is coalesced.
     * Note that each thread should read from a row in
     * shared memory. Since the data is stored shifted. For example, data [0,0]
     * is bank 0 and data [0,1] is in bank 1, so there is no bank conflict; thread
     * 0 and thread 1 is accessing different bank.
     */
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

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 32);
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
