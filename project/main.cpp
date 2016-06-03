#include <iostream>
#include <fstream>
#include <algorithm>
#include "matrix_utility.h"
#include "deconvolution.h"
#include <string.h>
#include <stdio.h>
using namespace std;

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

    // test matrix utility
    /*
    float mat[] = {1.0, 2.0, 3.0, 4.0};
    float det = matrix_utility::det(mat, 2);
    printf("det = %f \n", det);

    float a[] = {2.0, 4.0, 5.0, -1.0};
    float* result = new float[4];
    matrix_utility::convolve(a, mat, result, 4);
    printf("convolution result = %f %f %f %f\n", result[0],result[1],result[2],result[3]);
    return 0; */
    float* output = new float[size];
    dim_t input_dim = {w, h};
    run_deconvolution(input, output, input_dim, steps);
    std::string filename;
    filename = argv[1];
    std::size_t s = filename.find(".txt");
    filename.insert(s, "_deblur");
    std::ofstream outfile;
    outfile.open(filename);
    if(outfile.is_open()){
        for(int i; i<h*w - 1; i++){
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