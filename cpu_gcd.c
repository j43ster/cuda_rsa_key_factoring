#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string.h>

#include "pairwise_gcd.h"
#include "gmp_mp_helper.h"

#define MAX_SIZE 2
#define FILENAME "keys.txt"

using namespace std;

int main() {

   mp_int intlist[MAX_SIZE];
   compromised_keys comp_key_idxs;
   int size;
   int compromizable_pairs;
   char filename[1024];

   strcpy(filename, FILENAME);

   size = parse_largeint_file(filename, intlist, MAX_SIZE, true);

   printf("read %d ints from file %s\n", size, filename);

   compromizable_pairs = pairwise_gcd(intlist, size, &comp_key_idxs, true); 

   return 0;
}
