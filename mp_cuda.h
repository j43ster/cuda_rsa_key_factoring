#ifndef MP_H
#define MP_H

#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

#define MOST_SIG_BIT ((unsigned int)(1 << (sizeof(unsigned int)*8 - 1)))
#define LEAST_SIG_BIT 1

#define NUM_BITS 1024
#define NUM_WORDS (NUM_BITS/(8*sizeof(unsigned int)))

#define TRUE 1
#define FALSE 0

#define BLOCK_WIDTH 16
#define GRID_WIDTH 10

#define RES_SIZE (GRID_WIDTH*GRID_WIDTH*BLOCK_WIDTH*BLOCK_WIDTH)
#define RES_WIDTH (GRID_WIDTH*BLOCK_WIDTH)

#define MAX_SIZE 200000
#define FILE_MAX 4096

void verify_arguments(int argc, char** argv, char* filename);

static void HandleError( cudaError_t err, const char *file, int line){
    if (err != cudaSuccess) {
       printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line); 
       exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



// least significant word first
typedef struct mp_int {
   unsigned int idx[NUM_WORDS];
} mp_int;

typedef struct result_keys {

   int idx_a;
   int idx_b;

} result_keys;



int parse_largeint_file(char* filename, mp_int* intlist, int max_size, int verbose);

void print_mpz_hex(char* description, mpz_t num);
void print_mp_hex(char* description, mp_int* num);

void mp_import_mpz(mp_int* dest, mpz_t source);
void mp_export_mpz(mpz_t dest, mp_int* source);

__host__ __device__ void mp_init(mp_int* res);
__device__ void mp_int_copy(mp_int* dest, mp_int* source);

__host__ __device__ void mp_int_print_hex(mp_int* num);
__device__ void mp_int_gcd(mp_int* res, mp_int* a, mp_int* b);

__device__ void mp_int_sub(mp_int* res, mp_int* lhs, mp_int* rhs);
__device__ void mp_int_shift_left(mp_int* res);
__device__ void mp_int_shift_right(mp_int* res);

__device__ int mp_int_is_even(mp_int* num);
__device__ int mp_int_is_odd(mp_int* num);
__device__ int mp_int_is_zero(mp_int* num);

__device__ int mp_int_gt(mp_int* lhs, mp_int* rhs);
__device__ int mp_int_gte(mp_int* lhs, mp_int* rhs);
__device__ int mp_int_lt(mp_int* lhs, mp_int* rhs);
__device__ int mp_int_lte(mp_int* lhs, mp_int* rhs);
__device__ int mp_int_equal(mp_int* lhs, mp_int* rhs);

#endif
