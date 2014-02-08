#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string>

using namespace std;

void random_prime(mpz_t prime, unsigned long num_bits);
void print_mp(string prefix, mpz_t num);
