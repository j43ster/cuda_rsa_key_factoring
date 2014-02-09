#ifndef MP_CUDA_H
#define MP_CUDA_H

#include <stdio.h>
#include "mp_int.h"

__host__ __device__ void cu_mp_init(mp_int* res);
__host__ __device__ void cu_mp_int_copy(mp_int* dest, mp_int* source);

__device__ void cu_mp_int_print_hex(mp_int* num);
__device__ void cu_mp_int_gcd(mp_int* res, mp_int* a, mp_int* b);

__device__ void cu_mp_int_sub(mp_int* res, mp_int* lhs, mp_int* rhs);
__device__ void cu_mp_int_shift_left(mp_int* res);
__device__ void cu_mp_int_shift_right(mp_int* res);

__device__ int cu_mp_int_is_even(mp_int* num);
__device__ int cu_mp_int_is_odd(mp_int* num);
__device__ int cu_mp_int_is_zero(mp_int* num);

__device__ int cu_mp_int_gt(mp_int* lhs, mp_int* rhs);
__device__ int cu_mp_int_gte(mp_int* lhs, mp_int* rhs);
__device__ int cu_mp_int_lt(mp_int* lhs, mp_int* rhs);
__device__ int cu_mp_int_lte(mp_int* lhs, mp_int* rhs);
__device__ int cu_mp_int_equal(mp_int* lhs, mp_int* rhs);

#endif
