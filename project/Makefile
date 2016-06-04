CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
      LDFLAGS       := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS       := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

TARGETS = image_deblur
OBJS = deconvolution.o main.o matrix_utility.o cuda_matrix_utility_c.o cuda_matrix_utility.o

all: $(TARGETS)

image_deblur: $(OBJS)
	$(CC) -o image_deblur $(OBJS) $(LDFLAGS)

main.o: main.cpp
	$(CC) $(CCFLAGS) -std=c++11 -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -o $@ -c $<

deconvolution.o: deconvolution.cpp
	$(CC) $(CCFLAGS) -std=c++11 -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -o $@ -c $<

matrix_utility.o: matrix_utility.cpp
	$(CC) $(CCFLAGS) -std=c++11 -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -o $@ -c $<

cuda_matrix_utility_c.o: cuda_matrix_utility.cpp
	$(CC) $(CCFLAGS) -std=c++11 -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH) -o $@ -c $<

cuda_matrix_utility.o: cuda_matrix_utility.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<


clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)