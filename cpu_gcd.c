#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>

#include "pairwise_gcd.h"
#include "gmp_mp_helper.h"

#define MAX_SIZE 200000
#define FILE_MAX 4096

void verify_arguments(int argc, char** argv, char* filename);

int main(int argc, char** argv) {

   mp_int *intlist = malloc(sizeof(mp_int)*MAX_SIZE);
   compromised_keys comp_key_idxs;
    int size, compromizable_pairs;
   char filename[FILE_MAX];

   verify_arguments(argc, argv, filename);

   size = parse_largeint_file(filename, intlist, MAX_SIZE, 0);

   printf("read %d keys from file %s\n", size, filename);

   compromizable_pairs = pairwise_gcd(intlist, size, &comp_key_idxs, 1); 

   free(intlist);

   return 0;
}

void verify_arguments(int argc, char** argv, char* filename) {

   if (argc < 2) {
      fprintf(stderr, "invalid argument list, requires: %s filename\n", argv[0]);
      exit(0);
   }
   else {
      strncpy(filename, argv[1], FILE_MAX - 1);
   }
}
