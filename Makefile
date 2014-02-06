
default: gcd_cpu gcd_gpu

tests: mp_test mp_gmp_test

all: mp_test gcd_cpu gcd_gpu 

cpu: gcd_cpu

mp_test:
	gcc -o mp_test mp_test.c mp.c

mp_gmp_test:
	g++ -o mp_gmp_test gmp_helper.cpp gmp_mp_test.c mp.c gmp_mp_helper.c -lgmp

gcd_cpu:
	g++ -o gcd_cpu cpu_gcd.c mp.c pairwise_gcd.c gmp_mp_helper.c -lgmp

gcd_gpu:
	g++ -o part5 part5.cpp gmp_helper.cpp TextbookRSA.cpp pairwise_gcd.cpp -lgmp

clean:
	rm mp_test gcd_cpu gcd_gpu mp_gmp_test
