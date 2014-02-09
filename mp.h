#ifndef MP_H
#define MP_H

#include "mp_int.h"

void mp_init(mp_int* res);
void mp_int_copy(mp_int* dest, mp_int* source);

void mp_int_print_hex(mp_int* num);
void mp_int_gcd(mp_int* res, mp_int* a, mp_int* b);

void mp_int_sub(mp_int* res, mp_int* lhs, mp_int* rhs);
void mp_int_shift_left(mp_int* res);
void mp_int_shift_right(mp_int* res);

int mp_int_is_even(mp_int* num);
int mp_int_is_odd(mp_int* num);
int mp_int_is_zero(mp_int* num);

int mp_int_gt(mp_int* lhs, mp_int* rhs);
int mp_int_gte(mp_int* lhs, mp_int* rhs);
int mp_int_lt(mp_int* lhs, mp_int* rhs);
int mp_int_lte(mp_int* lhs, mp_int* rhs);
int mp_int_equal(mp_int* lhs, mp_int* rhs);

#endif
