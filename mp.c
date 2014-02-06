#include "mp.h"
#include <stdio.h>

void mp_init(mp_int* res) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      res->idx[i] = 0;
   }
}

void mp_int_copy(mp_int* dest, mp_int* source) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      dest->idx[i] = source->idx[i];
   }
}

void mp_int_print_hex(mp_int* num) {

   int i;
   int print_zero = 0;

   for (i = NUM_WORDS-1; i >= 0; i--) {
      if (num->idx[i] || print_zero) {
         printf("%.8x", num->idx[i]);
         print_zero = 1;
      }
   }
}

void mp_int_gcd(mp_int* res, mp_int* lhs, mp_int* rhs) {

   int i;
   int a_even, b_even;
   int done = FALSE;
   int num_shifts = 0;
   mp_int a, b;

   mp_int_copy(&a, lhs);
   mp_int_copy(&b, rhs);

   mp_init(res);
   res->idx[NUM_WORDS-1] = 1;

   int iteration = 0;

   while (!done) {

      //printf("on iteration: %d\n", iteration++);
      //printf("last words are: %u, %u\n", a.idx[0], b.idx[0]);


      if (mp_int_is_zero(&a) || mp_int_is_zero(&b))
         break;

      a_even = mp_int_is_even(&a);
      b_even = mp_int_is_even(&b);

      if (a_even && b_even) {
         num_shifts++;
         mp_int_shift_right(&a);
         mp_int_shift_right(&b);
      }
      else if (a_even && !b_even) {
         mp_int_shift_right(&a);
      }
      else if (!a_even && b_even) {
         mp_int_shift_right(&b);
      }
      else { // both are odd

         if (mp_int_equal(&a, &b)) {
            mp_int_copy(res, &a);
            done = TRUE;
         }
         else if (mp_int_lt(&a, &b)) {
            mp_int_sub(&b, &b, &a);
            mp_int_shift_right(&b);
         }
         else {
            mp_int_sub(&a, &a, &b);
            mp_int_shift_right(&a);
         }
      }
   }

   for (i = 0; i < num_shifts; i++) {
      mp_int_shift_left(res);
   }
}

void mp_int_sub(mp_int* res, mp_int* a, mp_int* b) {

   int i, j, borrow, current; 
   mp_int lhs, rhs;

   borrow = 0;

   mp_int_copy(&lhs, a);
   mp_int_copy(&rhs, b);

   //printf("NUM_WORDS is: %d\n", NUM_WORDS);

   for (i = NUM_WORDS - 1; i >= 0; i--) {

      //printf("idx: %d, lhs: %u, rhs %u\n", i, a->idx[i], b->idx[i]);

      if (lhs.idx[i] >= rhs.idx[i]) {
         res->idx[i] = lhs.idx[i] - rhs.idx[i];
      }
      else { // need to borrow
         j = i + 1;
         //printf("start borrow idx: %d\n", j);
         while (res->idx[j] == 0) {
            res->idx[j] = UINT_MAX;
            j++;
         }
         //printf("borrowing from index %d\n", j);
         res->idx[j] -= 1;

         res->idx[i] = UINT_MAX - rhs.idx[i];
         res->idx[i] += lhs.idx[i] + 1;
      }
   }
}

void mp_int_shift_left(mp_int* res) {

   int i;

   for (i = NUM_WORDS - 1; i >= 0; i--) {

      res->idx[i] = res->idx[i] << 1;
      
      if (i > 0 && res->idx[i-1] & MOST_SIG_BIT) {
         res->idx[i] += LEAST_SIG_BIT;
      }
   }
}

void mp_int_shift_right(mp_int* res) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {

      res->idx[i] = res->idx[i] >> 1;
      
      if (i < NUM_WORDS - 1 && res->idx[i+1] & LEAST_SIG_BIT) {
         res->idx[i] += MOST_SIG_BIT;
      }
   }
}

int mp_int_gt(mp_int* lhs, mp_int* rhs) {

   int i;

   for (i = NUM_WORDS - 1; i >= 0; i--) {
      if (lhs->idx[i] > rhs->idx[i]) {
         return TRUE;
      }
      else if (rhs->idx[i] > lhs->idx[i]) {
         return FALSE;
      }
   }

   return FALSE;
}

int mp_int_gte(mp_int* lhs, mp_int* rhs) {

   return (!mp_int_gt(rhs, lhs));
}

int mp_int_lt(mp_int* lhs, mp_int* rhs) {

   return mp_int_gt(rhs, lhs);
}

int mp_int_lte(mp_int* lhs, mp_int* rhs) {

   return (!mp_int_gt(lhs, rhs));
}

int mp_int_is_odd(mp_int* num) {

   return (num->idx[0] & 1);
}

int mp_int_is_even(mp_int* num) {

   return (!mp_int_is_odd(num));
}

int mp_int_is_zero(mp_int* num) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      if (num->idx[i]) {
         return FALSE;
      }
   }

   return TRUE;
}

int mp_int_equal(mp_int* a, mp_int* b) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      if (a->idx[i] != b->idx[i])
         return FALSE;
   }

   return TRUE;
}
