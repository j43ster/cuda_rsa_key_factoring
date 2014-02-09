#ifndef MP_H
#define MP_H

#include <limits.h>

#define MOST_SIG_BIT (1 << (sizeof(int)*8 - 1))
#define LEAST_SIG_BIT 1

#define NUM_BITS 1024
#define NUM_WORDS (NUM_BITS/(8*sizeof(unsigned int)))

#define TRUE 1
#define FALSE 0

// least significant word first
typedef struct mp_int {
   unsigned int idx[NUM_WORDS];
} mp_int;

typedef struct result_keys {

   int idx_a;
   int idx_b;

} result_keys;

__host__ __device__ void mp_init(mp_int* res);
__host__ __device__ void mp_int_copy(mp_int* dest, mp_int* source);

__device__ void mp_int_print_hex(mp_int* num);
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
