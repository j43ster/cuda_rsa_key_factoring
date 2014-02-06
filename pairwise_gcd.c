#include "pairwise_gcd.h"

int pairwise_gcd(mp_int* intlist, int size, compromised_keys* comp_key_idxs, int verbose) {

   int compd_key_count = 0;

   mp_int cf;
   mp_init(&cf);

   mp_int one;
   mp_init(&one);
   one.idx[NUM_WORDS-1] = 1;

   for (int i = 0; i < size; i++) {
      for (int j = i+1; j < size; j++) {

         printf("comparing keys <n%d, n%d>\n", i+1, j+1);

         mp_int_gcd(&cf, &intlist[i], &intlist[j]);
         /*if (mp_int_gt(&cf, &one)) { // if cf is greater than 1

            comp_key_idxs->idx_a[compd_key_count] = i;
            comp_key_idxs->idx_b[compd_key_count] = j;
            compd_key_count++;

            if (verbose) {
               //printf("numbers n%d and n%d share a gcd of ", i, j);
               //mpz_out_str(stdout, 10, cf);
               //printf("\n");
            }
         }*/
      }
   }

   return compd_key_count;
}
