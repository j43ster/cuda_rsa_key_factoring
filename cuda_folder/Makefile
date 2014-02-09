NVFLAGS= -O2 -G -g -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 
# list .c and .cu source files here



SRCFILES=mp.cu  


all:	mp_cuda	

mp_cuda: $(SRCFILES) 
	nvcc $(NVFLAGS) $(LIBRARIES) -o mp_cuda $^ 
double: $(SRCFILES)
	nvcc $(NVFLAGS) $(LIBRARIES) -o mp_cuda -DDOUBLE  $^
clean: 
	rm -f *.o mp_cuda

