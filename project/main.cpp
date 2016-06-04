#include <iostream>
#include <fstream>
#include <algorithm>
#include "matrix_utility.h"
#include "deconvolution.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

using namespace std;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


int main(int argc, char* argv[]) {
    if (argc != 5){
        std::cerr << "invalid arguement: [input filename] [max step] [width] [height] " << std::endl;
        return -1;
    }

    FILE *fp;
    int w = atoi(argv[3]);
    int h = atoi(argv[4]);
    int steps = atoi(argv[2]);
    int size = w *h;

    std::string line;
    char* cstr = new char[256];
    float* input = new float[size];
    fstream infile(argv[1]);
    if(infile.is_open()) {
        int i = 0;
        while (infile.getline(cstr, size, ',')) {
            int k = atoi(cstr);
            input[i] = float(k);
            i++;
            //std::cout << k << std::endl;
        }
    }
    else{
        std::cerr<<"fail to open file\n" << std::endl;
        return -1;
    }
    infile.close();

    float* output = new float[size];
    dim_t input_dim = {w, h};

    // use cuda machinary to time
    cudaEvent_t start;
    cudaEvent_t stop;
    float cpu_time_ms = -1;
    float gpu_time_ms = -1;

#define START_TIMER() {                         \
      gpuErrchk(cudaEventCreate(&start));       \
      gpuErrchk(cudaEventCreate(&stop));        \
      gpuErrchk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrchk(cudaEventRecord(stop));                     \
      gpuErrchk(cudaEventSynchronize(stop));                \
      gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrchk(cudaEventDestroy(start));                   \
      gpuErrchk(cudaEventDestroy(stop));                    \
    }

// and start timer
    START_TIMER();
    run_deconvolution_cpu(input, output, input_dim, steps);
    STOP_RECORD_TIMER(cpu_time_ms);
    printf("CPU deconvolution finish in %f ms\n", cpu_time_ms);

    START_TIMER();
    run_deconvolution_gpu(input, output, input_dim, steps);
    STOP_RECORD_TIMER(gpu_time_ms);
    printf("GPU deconvolution finish in %f ms\n", gpu_time_ms);

    std::string filename;
    filename = argv[1];
    std::size_t s = filename.find(".txt");
    filename.insert(s, "_deblur");
    std::ofstream outfile;
    outfile.open(filename);
    if(outfile.is_open()){
        for(int i = 0; i<h*w - 1; i++){
            outfile<<(int)output[i]<<",";
        }
        outfile<<(int)output[h*w - 1];
        outfile.close();
        std::cout<<"write to file " <<filename << std::endl;
    } else{
        std::cerr<<"Fail to open file" << filename << std::endl;
        return -1;
    }
    return 0;
}