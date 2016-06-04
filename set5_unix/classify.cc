#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"
#include "ta_utilities.hpp"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
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


// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
    gpuErrChk(cudaEventCreate(&start));         \
    gpuErrChk(cudaEventCreate(&stop));          \
    gpuErrChk(cudaEventRecord(start));          \
}

#define STOP_RECORD_TIMER(name) {                           \
    gpuErrChk(cudaEventRecord(stop));                       \
    gpuErrChk(cudaEventSynchronize(stop));                  \
    gpuErrChk(cudaEventElapsedTime(&name, start, stop));    \
    gpuErrChk(cudaEventDestroy(start));                     \
    gpuErrChk(cudaEventDestroy(stop));                      \
}

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

/*
void CUDART_CB CheckOutputCallback(cudaStream_t stream, cudaError_t status, void *data){
    printf("Inside callback %d\n", (size_t)data);
}
*/


// Fills output with standard normal data
void gaussianFill(float *output, int size) {
    // seed generator to 2015
    std::default_random_engine generator(2015);
    std::normal_distribution<float> distribution(0.0, 0.1);
    for (int i=0; i < size; i++) {
        output[i] = distribution(generator);
    }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) {
    stringstream stream(review_str);
    int component_idx = 0;

    for (string component; getline(stream, component, ','); component_idx++) {
        output[stride * component_idx] = atof(component.c_str());
    }
    assert(component_idx == REVIEW_DIM + 1);
}

void classify(istream& in_stream, int batch_size) {

    float* host_weight;
    float* dev_weight;
    int step_size = 1;
    float* host_inputs[2];
    float* dev_inputs[2];

    host_weight = (float*)malloc(sizeof(float) * REVIEW_DIM);
    host_inputs[0] = (float*)malloc(sizeof(float) * (REVIEW_DIM + 1) * batch_size);
    host_inputs[1] = (float*)malloc(sizeof(float) * (REVIEW_DIM + 1) * batch_size);
    gaussianFill(host_weight, REVIEW_DIM);

    cudaMalloc(&dev_weight, sizeof(float) * REVIEW_DIM);
    checkCUDAKernelError();
    cudaMalloc(&(dev_inputs[1]), sizeof(float) * (REVIEW_DIM + 1) * batch_size);
    cudaMalloc(&(dev_inputs[0]), sizeof(float) * (REVIEW_DIM + 1) * batch_size);
    cudaMemcpy(dev_weight, host_weight, sizeof(float) * REVIEW_DIM, cudaMemcpyHostToDevice);
    checkCUDAKernelError();

    cudaStream_t s[2];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);



    int review_idx = 0;
    int batch_count = 0;
    int buffer_num = 0;
    int prev_buffer_num = 0;
    int batch_num = 0;

    int first_batch = 1;
    for (string review_str; getline(in_stream, review_str); review_idx++) {
        // Read data from file until we fill up a batch
        readLSAReview(review_str, (host_inputs[buffer_num]) + batch_count * (REVIEW_DIM + 1), 1);

        batch_count++;
        if (batch_count == batch_size){
            // syncronize previous buffer to make sure that the previous kernal is completed
            cudaStreamSynchronize(s[buffer_num]);
            // copy the new batch to device
            cudaMemcpyAsync(dev_inputs[buffer_num], host_inputs[buffer_num], sizeof(float)*(REVIEW_DIM + 1) * batch_size,
                            cudaMemcpyHostToDevice,s[buffer_num]);

            if (!first_batch){
                // run kernal on the previous data that was copied to device
                cudaClassify(dev_inputs[prev_buffer_num], batch_size, step_size, dev_weight,s[prev_buffer_num]);
            }

            first_batch = 0;
            prev_buffer_num = buffer_num;
            buffer_num = (buffer_num + 1) % 2;
            batch_count = 0;
            batch_num++;
            //cudaStreamSynchronize(s[prev_buffer_num]);
           // printf("batch %d, err = %f \n",batch_num, err);
        }
    }

    cudaStreamSynchronize(s[buffer_num]);
    // need to run the last batch
    cudaClassify(dev_inputs[prev_buffer_num], batch_size, step_size, dev_weight, s[prev_buffer_num]);
    cudaStreamSynchronize(s[prev_buffer_num]);


    cudaMemcpy(host_weight, dev_weight, sizeof(float) * REVIEW_DIM, cudaMemcpyDeviceToHost);

    printf("final weight : \n");
    for (int i = 0; i < REVIEW_DIM; i++){
        printf("%f \n", host_weight[i]);
    }

    cudaFree(dev_weight);
    cudaFree(dev_inputs[1]);
    cudaFree(dev_inputs[0]);
    free(host_inputs[0]);
    free(host_inputs[1]);
}

int main(int argc, char** argv) {
    if (argc != 2) {
		printf("./classify <path to datafile>\n");
		return -1;
    } 
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 100;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
	
    // Init timing
	float time_initial, time_final;
	
    int batch_size = 16384;
	
	// begin timer
	time_initial = clock();
	
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);
	
	// End timer
	time_final = clock();
	printf("Total time to run classify: %f (s)\n", (time_final - time_initial) / CLOCKS_PER_SEC);
	

}