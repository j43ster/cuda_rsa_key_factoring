#include <stdio.h>
#include <gmp.h>
#include <time.h>

void random_prime(mpz_t prime, unsigned long num_bits);
void print_mp(char* prefix, mpz_t num);
