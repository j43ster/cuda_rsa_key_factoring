//#ifndef GMP_MP_HELPER_H
//#define GMP_MP_HELPER_H

#include <stdio.h>
#include <gmp.h>
#include <assert.h>
#include "mp.h"

int parse_largeint_file(char* filename, mp_int* intlist, int max_size, int verbose);

void print_mpz_hex(char* description, mpz_t num);
void print_mp_hex(char* description, mp_int* num);

void mp_import_mpz(mp_int* dest, mpz_t source);
void mp_export_mpz(mpz_t dest, mp_int* source);

//#endif
