CXX       = g++
NVCC      = nvcc
ARCH      = sm_35
CUDAFLAGS = -arch=$(ARCH) -Xcompiler -fopenmp -O3

all: a2

a2: a2.cu
	$(NVCC) $(CUDAFLAGS) -o a2 a2.cu

clean:
	rm -f a2 knn.txt approx_knn.txt kmeans.txt

.PHONY: all clean