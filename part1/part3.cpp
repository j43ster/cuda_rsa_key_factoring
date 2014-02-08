#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string>
#include <stdlib.h>
#include <string.h>

#include "pairwise_gcd.h"

#define MAX_SIZE 200000
#define FILE_MAX 4096

using namespace std;

void verify_arguments(int argc, char** argv, char* filename) {

   if (argc < 2) {
      fprintf(stderr, "invalid argument list, requires: %s filename\n", argv[0]);
      exit(0);
   }
   else {
      strncpy(filename, argv[1], FILE_MAX - 1);
   }
}

int main(int argc, char** argv) {

   mpz_t intlist[MAX_SIZE];
   int size;
   char filename[FILE_MAX];

   verify_arguments(argc, argv, filename);

   size = parse_largeint_file(filename, intlist, MAX_SIZE, false);
   pairwise_gcd(intlist, size, true); 

   return 0;
}
