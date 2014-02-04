
default: gcd_cpu gcd_gpu

tests: mp_test

all: mp_test gcd_cpu gcd_gpu 

mp_test:
	gcc -o mp_test mp_test.c mp.c

gcd_cpu:
	g++ -o gcd_cpu part5.cpp gmp_helper.cpp TextbookRSA.cpp pairwise_gcd.cpp -lgmp

gcd_gpu:
	g++ -o part5 part5.cpp gmp_helper.cpp TextbookRSA.cpp pairwise_gcd.cpp -lgmp

clean:
	rm mp_test gcd_cpu gcd_gpu 
