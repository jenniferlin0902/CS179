
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)

Modified by Jordan Bonilla and Matthew Cedeno (2016)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>
#include <time.h>
#include <math.h>
#include "ta_utilities.hpp"

#define PI 3.14159265358979


/* texture reference declaration */
texture<float, cudaTextureType1D, cudaReadModeElementType> sinogramTextureRef;

/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}


__global__
void cudaHighPassFilterKernal(cufftComplex *sinogram_input, uint sinogram_width, uint nAngles){
    float step = 2.0/(float)sinogram_width;
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    while (threadId < (sinogram_width*nAngles)){
        float scale = 2.0 - step * (float)(threadId % sinogram_width);
        scale = (scale > 1) ? 2.0 - scale : scale;
        sinogram_input[threadId].x = sinogram_input[threadId].x * scale;
        sinogram_input[threadId].y = sinogram_input[threadId].y * scale;
        threadId += blockDim.x * gridDim.x;
    }
}

__global__
void cudaComplex2FloatKernal(cufftComplex* sinogram_complex, float* sinogram_float, uint size){
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    while (threadId < size){
        sinogram_float[threadId] = sinogram_complex[threadId].x;
        threadId += blockDim.x * gridDim.x;
    }
}


__global__
void cudaXrayReconstruction(float* sinogram_input,float* output, uint nAngles, uint sinogram_width, uint height, uint width){
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    while (threadId < height * width){
        int y = -(threadId / width - height/2);
        int x = threadId % width - width/2;
        for (uint angle = 0; angle < nAngles; angle++){
            float theta = angle*PI/nAngles;

            int d;
            float q, x_i, y_i, m;

            if (theta == 0){
                d = x;
            }else if (theta == PI/2){
                d = y;
            }else{
                m = -1.0 * cos(theta)/sin(theta);
                q = -1.0/m;
                x_i = ((float)y - m * (float)x)/(float)(q - m);
                y_i = q*x_i;
                d = (int)(sqrt(x_i * x_i + y_i * y_i) + 0.5);

                if (((q > 0) && (x_i < 0)) || ((q < 0) && (x_i > 0))){
                    d = -d;
                }
            }
            uint abs_d = (d < 0) ? -d : d;
            if (abs_d < sinogram_width/2){
                output[threadId] += sinogram_input[angle*sinogram_width + sinogram_width/2 + d];
            }
        }
        threadId += blockDim.x * gridDim.x;
    }
}

void cudaCallXrayReconstruction(const unsigned int blocks,
                                const unsigned int threadsPerBlock,
                                float* sinogram_input,
                                float* output,
                                uint nAngles,
                                uint sinogram_width,
                                uint height,
                                uint width){
    cudaXrayReconstruction<<<blocks, threadsPerBlock>>>(sinogram_input, output, nAngles, sinogram_width, height, width);
}

void cudaCallHighPassFilterKernal(const unsigned int blocks,
                                   const unsigned int threadsPerBlock,
                                   cufftComplex* sinogram_input,
                                   uint sinogram_width,
                                    uint nAngles){
    cudaHighPassFilterKernal<<<blocks, threadsPerBlock>>>(sinogram_input, sinogram_width, nAngles);
}



void cudaCallComplex2FloatKernal(const unsigned int blocks,
                                const unsigned int threadsPerBlock,
                                cufftComplex* sinogram_complex,
                                float* sinogram_float,
                                uint size){
    cudaComplex2FloatKernal<<<blocks, threadsPerBlock>>>(sinogram_complex, sinogram_float, size);
}

int main(int argc, char** argv){
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 10;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Begin timer and check for the correct number of inputs
    time_t start = clock();
    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Input sinogram text file's name > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output text file's name >\n");
        exit(EXIT_FAILURE);
    }






    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* TODO: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    cudaMalloc(&dev_sinogram_cmplx, sizeof(cufftComplex) * sinogram_width * nAngles);
    cudaMalloc(&dev_sinogram_float, sizeof(float) * sinogram_width *nAngles);
    cudaMalloc(&output_dev, size_result);
    cudaMemcpy(dev_sinogram_cmplx, sinogram_host, sizeof(cufftComplex) * sinogram_width * nAngles, cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_sinogram_float, sino)
    cudaMemset(dev_sinogram_float, 0x0, sizeof(float) * sinogram_width * nAngles);
    cudaMemset(output_dev, 0x0, size_result);

    /* TODO 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */

    cufftHandle plan1;

    int batch = nAngles;
    cufftPlan1d(&plan1, sinogram_width, CUFFT_C2C, batch);
    //cufftPlan1d(&plan2, sinogram_width, CUFFT_R2C, batch);
    cufftExecC2C(plan1, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);
    cudaCallHighPassFilterKernal(nBlocks, threadsPerBlock, dev_sinogram_cmplx,sinogram_width, nAngles);
    checkCUDAKernelError();
    cufftExecC2C(plan1, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE);
    cudaCallComplex2FloatKernal(nBlocks, threadsPerBlock, dev_sinogram_cmplx, dev_sinogram_float, sinogram_width*nAngles);
    checkCUDAKernelError();



    cufftDestroy(plan1);
 //   cufftDestroy(plan2);
    cudaFree(dev_sinogram_cmplx);



    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    /*
    cudaArray* cArraySinogram;

    textureReference* texRefPtr;
    cudaGetTextureReference(&texRefPtf, &sinogramTextureRef);
    cudaChannelFormatDesc channelDesc;
    channelDesc= cudaCreateChannelDesc<float>();

    cudaBindTexture(0, texRefPtr, dev_sinogram_float, &channelDesc, sinogram_width * nAngles * sizeof(float));

    */
    cudaCallXrayReconstruction(nBlocks, threadsPerBlock, dev_sinogram_float, output_dev,nAngles, sinogram_width,height, width);
    checkCUDAKernelError();
    cudaMemcpy(output_host, output_dev, size_result, cudaMemcpyDeviceToHost);
    cudaFree(output_dev);



    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);
    printf("CT reconstruction complete. Total run time: %f seconds\n", (float) (clock() - start) / 1000.0);
    return 0;
}




