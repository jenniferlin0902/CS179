/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#ifndef CUDA_1D_FD_WAVE_CUDA_CUH
#define CUDA_1D_FD_WAVE_CUDA_CUH


/* TODO: This is a CUDA header file.
If you have any functions in your .cu file that need to be
accessed from the outside, declare them here */
void callCuda1DWaveBCKernal(
        float* dev_new,
        float left_bc,
        uint numberOfNodes,
        uint threadPerBlock,
        uint blocks);

void callCuda1DFDWaveKernal(
        float* dev_new,
        float* dev_current,
        float* dev_old,
        float courantSquared,
        uint numberOfNodes,
        uint threadPerBlock,
        uint blocks);
#endif // CUDA_1D_FD_WAVE_CUDA_CUH
