#include "gmp_mp_helper.h"

int parse_largeint_file(char* filename, mp_int* intlist, int max_size, int verbose) {

   FILE* file = fopen(filename, "r");

   mpz_t tmp;
   mpz_init(tmp);

   int i = 0;

   while (mpz_inp_str(tmp, file, 10) > 0 && i < max_size) {

      if (verbose) {
         printf("n%d = ", i+1);
         mpz_out_str(stdout, 10, tmp);
         printf("\n\n");
      }

      mp_init(&intlist[i]);
      mp_import_mpz(&intlist[i], tmp);

      i++;
   }

   fclose(file);

   return i;
}

void print_mpz_hex(char* description, mpz_t num) {

   printf("%s", description);
   mpz_out_str(stdout, 16, num);
   printf("\n");
}

void print_mp_hex(char* description, mp_int* num) {

   printf("%s", description);
   mp_int_print_hex(num);
   printf("\n");
}

void mp_import_mpz(mp_int* dest, mpz_t source) {

   size_t count;

   mpz_export(&dest->idx[0], &count, -1, sizeof(unsigned int), 0, 0, source);
}

void mp_export_mpz(mpz_t dest, mp_int* source) {

   mpz_import(dest, NUM_WORDS, -1, sizeof(unsigned int), 0, 0, &source->idx[0]);
}
