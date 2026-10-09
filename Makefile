MPICXX ?= mpicxx
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_70
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -fopenmp
NVCCFLAGS ?= -O2 -std=c++17 -arch=$(CUDA_ARCH)

.PHONY: all cpu gpu clean
all: cpu

cpu: program-cpu
gpu: program-gpu

program-cpu: tfidfstg.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

tfidfstg-gpu.o: tfidfstg.cpp CudaCosineSimilarity.hpp
	$(MPICXX) $(CXXFLAGS) -DUSE_CUDA -c $< -o $@

CudaCosineSimilarity.o: CudaCosineSimilarity.cu CudaCosineSimilarity.hpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

program-gpu: tfidfstg-gpu.o CudaCosineSimilarity.o
	$(MPICXX) $(CXXFLAGS) $^ -L$(CUDA_HOME)/lib64 -lcudart -o $@

clean:
	rm -f program-cpu program-gpu tfidfstg-gpu.o CudaCosineSimilarity.o
