default: gcd_cpu gcd_gpu gcd_gmp_cpu

tests: mp_test mp_gmp_test

all: mp_test gcd_cpu gcd_gpu 

cpu: gcd_cpu

CPU_SRC=cpu_gcd.c mp.c pairwise_gcd.c gmp_mp_helper.c gmp_helper.c
CFLAGS=-m64 -O2

GPU_C_SRC=mp.c pairwise_gcd.c gmp_mp_helper.c gmp_helper.c
GPU_SRC=gpu_gcd.cu 

mp_test:
	gcc -o mp_test mp_test.c mp.c

mp_gmp_test:
	gcc -o mp_gmp_test gmp_helper.c gmp_mp_test.c mp.c gmp_mp_helper.c -I /home/clupo/gmp/include -lgmp -L/home/clupo/gmp/lib

gcd_gmp_cpu:
	$(MAKE) -C ./part1/ part3
	cp ./part1/part3 ./gcd_cpu_gmp

gcd_cpu:
	gcc -c $(GPU_C_SRC) -I/home/clupo/gmp/include/ 
	gcc $(CFLAGS) -o gcd_cpu cpu_gcd.c *.o -I /home/clupo/gmp/include /home/clupo/gmp/lib/libgmp.a 
	#gcc $(CFLAGS) -o gcd_cpu $(CPU_SRC) -I /home/clupo/gmp/include /home/clupo/gmp/lib/libgmp.a 

gcd_gpu: 
	#/usr/local/cuda-5.5/bin/nvcc -O2 -rdc=true -o gcd_gpu -I/home/clupo/gmp/include/ -L/home/clupo/gmp/lib/ -lgmp /home/clupo/gmp/lib/libgmp.a $(GPU_SRC) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 $^  
	/usr/local/cuda-5.5/bin/nvcc -O2 -rdc=true -o gcd_gpu -I/home/clupo/gmp/include/ /home/clupo/gmp/lib/libgmp.a $(GPU_SRC) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 $^  

clean: clean_p1
	rm -f *.o mp_test gcd_cpu gcd_cpu_gmp gcd_gpu mp_gmp_test

clean_p1:
	$(MAKE) -C ./part1/ clean
