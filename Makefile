all:
	nvcc -arch=sm_50 -o cudasim.o -c CudaCosineSimilarity.cu
	mpicc -c tfidfstg.cpp -o main.o -lm -fopenmp
	mpiCC main.o cudasim.o -L/usr/local/cuda/lib64 -lcuda -lcudart -o program -lm -fopenmp -lstdc++


	
clean:





