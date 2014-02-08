#include "gmp_helper.h"

gmp_randstate_t randstate;
bool rand_initialized = false;

void init_rand() {

   gmp_randinit_mt(randstate);
   gmp_randseed_ui(randstate, time(0));

   rand_initialized = true;
}

void random_prime(mpz_t prime, unsigned long num_bits) {

   if (!rand_initialized) {
      init_rand();
   }

   mpz_urandomb(prime, randstate, num_bits);
   mpz_nextprime(prime, prime);
}

void print_mp(string prefix, mpz_t num) {

   printf("%s", prefix.c_str());
   mpz_out_str(stdout, 10, num);
   printf("\n");

}
