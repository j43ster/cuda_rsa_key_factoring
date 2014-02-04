#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string>

#include "pairwise_gcd.h"

#define MAX_SIZE 100

using namespace std;

int main() {

   mpz_t intlist[MAX_SIZE];
   int size;

   size = parse_largeint_file("largeints.txt", intlist, MAX_SIZE, true);
   pairwise_gcd(intlist, size, true); 

   return 0;
}
