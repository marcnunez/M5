all:
	nvcc conv_ex2.cu -o conv -I $(CUDA_HOME)/include/ -I $(CUDA_HOME)/samples/7_CUDALibraries/common/FreeImage/include/ -L $(CUDA_HOME)/lib64/ -L $(CUDA_HOME)/samples/7_CUDALibraries/common/FreeImage/lib/linux/x86_64/ -lcudnn -lfreeimage -std=c++11
clean:
	rm conv
