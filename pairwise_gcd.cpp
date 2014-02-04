#include "pairwise_gcd.h"

vector<pair<int, int> > pairwise_gcd(mpz_t* intlist, int size, bool verbose) {

   vector<pair<int, int> > broken_keys;

   mpz_t cf;
   mpz_init(cf);

   mpz_t one;
   mpz_init(one);
   mpz_set_ui(one, 1);

   for (int i = 0; i < size; i++) {
      for (int j = i+1; j < size; j++) {
         mpz_gcd(cf, intlist[i], intlist[j]);
         if (mpz_cmp(cf, one)) { // if cf is greater than 1

            broken_keys.push_back(pair<int, int>(i, j)); 

            if (verbose) {
               printf("numbers n%d and n%d share a gcd of ", i, j);
               mpz_out_str(stdout, 10, cf);
               printf("\n");
            }
         }
      }
   }

   return broken_keys;
}

int parse_largeint_file(string filename, mpz_t* int_list, int max_size, bool verbose) {

   FILE* file = fopen(filename.c_str(), "r");

   mpz_t t;
   mpz_init(t);

   int i = 0;

   while (mpz_inp_str(t, file, 10) > 0 && i < max_size) {

      if (verbose) {
         printf("n%d = ", i+1);
         mpz_out_str(stdout, 10, t);
         printf("\n\n");
      }

      mpz_init(int_list[i]);
      mpz_set(int_list[i], t);

      i++;
   }

   return i;
}

