#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string>
#include <vector>
#include <utility>
#include <stdlib.h>

#include "TextbookRSA.h"
#include "gmp_helper.h"
#include "pairwise_gcd.h"

using namespace std;

#define MAX_SIZE 10000

void break_rsa_key(mpz_t n, mpz_t p);
void read_large_int(char* filename, mpz_t i);

int main(int argc, char** argv) {

   mpz_t intlist[MAX_SIZE];
   int size;

   if (argc <2) {
      fprintf(stderr, "requires input file\n");
      exit(-1);
   }

   size = parse_largeint_file(argv[1], intlist, MAX_SIZE, false);

   mpz_t n1, n2, gcd, p;

   mpz_init(n1);
   mpz_init(n2);
   mpz_init(gcd);
   mpz_init(p);

   char text[1024];
   size_t count;

   for (int i = 0; i < size; i+=2) {

      mpz_set(n1, intlist[i]);
      mpz_set(n2, intlist[i+1]); 
      mpz_gcd(gcd, n1, n2);
    
      //print_mp("gcd is: ", gcd);

      mpz_set(p, gcd);
      break_rsa_key(n1, p);

      mpz_set(p, gcd); 
      break_rsa_key(n2, p); 
 
      /*break_encryption(pt1, ct, n1, p); 
      break_encryption(pt2, ct, n2, p);

      mpz_export(text, &count, 1, 1, 0, 0, pt1);
      text[count] = '\0';
      print_mp("using public key: ", n1);
      printf("decrytped message is: %s\n", text);

      printf("\n");

      mpz_export(text, &count, 1, 1, 0, 0, pt2);
      text[count] = '\0';
      print_mp("using public key: ", n2);
      printf("decrytped message is: %s\n", text);*/
   }
}

void break_rsa_key(mpz_t n, mpz_t p) {

   mpz_t q;

   mpz_init(q);

   mpz_divexact(q, n, p);
   TextbookRSA rsa(p, q);

   // print results
   mpz_out_str(stdout, 10, n);
   print_mp(":", rsa.m_d);
}

void read_large_int(char* filename, mpz_t i) {

   FILE* file = fopen(filename, "r");

   mpz_inp_str(i, file, 10);

   fclose(file);
}

