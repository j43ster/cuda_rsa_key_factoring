
default: gcd_cpu gcd_gpu gcd_gmp_cpu

tests: mp_test mp_gmp_test

all: mp_test gcd_cpu gcd_gpu 

cpu: gcd_cpu

mp_test:
	gcc -o mp_test mp_test.c mp.c

mp_gmp_test:
	gcc -o mp_gmp_test gmp_helper.c gmp_mp_test.c mp.c gmp_mp_helper.c -I /home/clupo/gmp/include -L/home/clupo/gmp/lib -lgmp

gcd_gmp_cpu:
	$(MAKE) -C ./part1/ part3

gcd_cpu:
	gcc -o gcd_cpu -O2 cpu_gcd.c mp.c pairwise_gcd.c gmp_mp_helper.c gmp_helper.c -I /home/clupo/gmp/include -L/home/clupo/gmp/lib -lgmp

gcd_gpu:
	true

clean: clean_p1
	rm -f mp_test gcd_cpu gcd_gpu mp_gmp_test

clean_p1:
	$(MAKE) -C ./part1/ clean
