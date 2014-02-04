#include <stdio.h>
#include <assert.h>

#include "mp.h"

int gt_test() {

   mp_int a, b;

   mp_init(&a);
   mp_init(&b);

   assert(mp_int_gt(&a, &b) == 0);
   assert(mp_int_gt(&b, &a) == 0);

   a.idx[NUM_WORDS-1] = 10;
   b.idx[NUM_WORDS-1] = 15;

   assert(mp_int_gt(&a, &b) == 0);
   assert(mp_int_gt(&b, &a) == 1);

   a.idx[0] = 1;

   assert(mp_int_gt(&a, &b) == 1);
   assert(mp_int_gt(&b, &a) == 0);
}

int gte_test() { 

   mp_int a, b;

   mp_init(&a);
   mp_init(&b);

   assert(mp_int_gte(&a, &b) == 1);
   assert(mp_int_gte(&b, &a) == 1);

   a.idx[NUM_WORDS-1] = 10;
   b.idx[NUM_WORDS-1] = 15;

   assert(mp_int_gte(&a, &b) == 0);
   assert(mp_int_gte(&b, &a) == 1);

   a.idx[0] = 1;

   assert(mp_int_gte(&a, &b) == 1);
   assert(mp_int_gte(&b, &a) == 0);
}

int lt_test() { // way greater

   mp_int a, b;

   mp_init(&a);
   mp_init(&b);

   assert(mp_int_lt(&a, &b) == 0);
   assert(mp_int_lt(&b, &a) == 0);

   a.idx[NUM_WORDS-1] = 10;
   b.idx[NUM_WORDS-1] = 15;

   assert(mp_int_lt(&a, &b) == 1);
   assert(mp_int_lt(&b, &a) == 0);

   a.idx[0] = 1;

   assert(mp_int_lt(&a, &b) == 0);
   assert(mp_int_lt(&b, &a) == 1);
}

int lte_test() { // way greater

   mp_int a, b;

   mp_init(&a);
   mp_init(&b);

   assert(mp_int_lte(&a, &b) == 1);
   assert(mp_int_lte(&b, &a) == 1);

   a.idx[NUM_WORDS-1] = 10;
   b.idx[NUM_WORDS-1] = 15;

   assert(mp_int_lte(&a, &b) == 1);
   assert(mp_int_lte(&b, &a) == 0);

   a.idx[0] = 1;

   assert(mp_int_lte(&a, &b) == 0);
   assert(mp_int_lte(&b, &a) == 1);
}

void shift_left_test() {

   mp_int a;

   mp_init(&a);

   mp_int_shift_left(&a);
   
   assert(a.idx[NUM_WORDS-1] == 0);

   a.idx[NUM_WORDS-1] = 2;
   mp_int_shift_left(&a);

   assert(a.idx[NUM_WORDS-1] == 4);
    
   a.idx[NUM_WORDS-1] += MOST_SIG_BIT;
   mp_int_shift_left(&a);

   assert(a.idx[NUM_WORDS-1] == 8);
   assert(a.idx[NUM_WORDS-2] == 1);
}

void shift_right_test() {

   mp_int a;

   mp_init(&a);

   mp_int_shift_right(&a);
   
   assert(a.idx[NUM_WORDS-1] == 0);

   a.idx[NUM_WORDS-1] = 2;
   mp_int_shift_right(&a);

   assert(a.idx[NUM_WORDS-1] == 1);
    
   a.idx[NUM_WORDS-2] += 1;
   a.idx[NUM_WORDS-2] += 8;
   mp_int_shift_right(&a);

   assert(a.idx[NUM_WORDS-1] == MOST_SIG_BIT);
   assert(a.idx[NUM_WORDS-2] == 4);
}

void sub_test() {

   mp_int a, b, res;

   mp_init(&a);
   mp_init(&b);
   mp_init(&res);


   a.idx[NUM_WORDS-1] = 61;
   a.idx[NUM_WORDS-2] = 8;
   b.idx[NUM_WORDS-1] = 10;

   mp_int_sub(&res, &a, &b);

   assert(res.idx[NUM_WORDS-1] == 51);
   assert(res.idx[NUM_WORDS-2] == 8);

   a.idx[NUM_WORDS-1] = 61;
   a.idx[NUM_WORDS-2] = 8;
   b.idx[NUM_WORDS-1] = UINT_MAX;
  
   mp_int_sub(&res, &a, &b);
 
   assert(res.idx[NUM_WORDS-1] == 62);
   assert(res.idx[NUM_WORDS-2] == 7);

   a.idx[NUM_WORDS-1] = 61;
   a.idx[NUM_WORDS-2] = 0;
   a.idx[NUM_WORDS-3] = 8;

   mp_int_sub(&res, &a, &b);
 
   assert(res.idx[NUM_WORDS-1] == 62);
   assert(res.idx[NUM_WORDS-2] == UINT_MAX);
   assert(res.idx[NUM_WORDS-3] == 7);
}

void is_even_test() {

   mp_int a;

   mp_init(&a);

   a.idx[NUM_WORDS-1] = 1;

   assert(mp_int_is_even(&a) == 0);

   a.idx[NUM_WORDS-1] = 2;

   assert(mp_int_is_even(&a) == 1);

   a.idx[NUM_WORDS-2] = 1;

   assert(mp_int_is_even(&a) == 1);

   a.idx[NUM_WORDS-2] = 2;

   assert(mp_int_is_even(&a) == 1);
}

void is_odd_test() {

   mp_int a;

   mp_init(&a);

   a.idx[NUM_WORDS-1] = 1;

   assert(mp_int_is_odd(&a) == 1);

   a.idx[NUM_WORDS-1] = 2;

   assert(mp_int_is_odd(&a) == 0);

   a.idx[NUM_WORDS-2] = 1;

   assert(mp_int_is_odd(&a) == 0);
}

void copy_test() {

   mp_int a, b;

   mp_init(&a);
   mp_init(&b);

   a.idx[NUM_WORDS-1] = 100;
   mp_int_copy(&b, &a);

   assert(b.idx[NUM_WORDS-1] == 100);

   b.idx[0] = 12;
   b.idx[NUM_WORDS-1] = 8;
   mp_int_copy(&a, &b);

   assert(a.idx[0] == 12);
   assert(a.idx[NUM_WORDS-1] == 8);
}

void gcd_test() {

   mp_int a, b, c;

   mp_init(&a);
   mp_init(&b);
   mp_init(&c);

   a.idx[NUM_WORDS-1] = 42;
   b.idx[NUM_WORDS-1] = 56;

   mp_int_gcd(&c, &a, &b);

   assert(c.idx[NUM_WORDS-1] == 14);
}

int main(void) {

   gcd_test();
   copy_test();

   gt_test();
   gte_test();
   lt_test();
   lte_test();

   shift_left_test();
   shift_right_test();

   sub_test();

   is_even_test();
   is_odd_test();
}
