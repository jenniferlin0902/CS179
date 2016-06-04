cmake_minimum_required(VERSION 3.3)
project(matrix_factorize)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp deconvolution.cpp deconvolution.h matrix_utility.cpp matrix_utility.h)
add_executable(matrix_factorize ${SOURCE_FILES})