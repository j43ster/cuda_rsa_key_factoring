#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string>
#include <vector>
#include <utility>

#include "TextbookRSA.h"
#include "gmp_helper.h"
#include "pairwise_gcd.h"

using namespace std;

#define MAX_SIZE 100

void break_encryption(mpz_t plaintext, mpz_t ciphertext, mpz_t n, mpz_t p);
void read_large_int(char* filename, mpz_t i);

int main(void) {

   mpz_t intlist[MAX_SIZE];
   int size;
   vector<pair<int, int> > compromised_keys;

   size = parse_largeint_file("largeints.txt", intlist, MAX_SIZE, false);
   compromised_keys = pairwise_gcd(intlist, size, false);

   mpz_t n1, n2, p, ct, pt1, pt2;

   mpz_init(n1);
   mpz_init(n2);
   mpz_init(p);
   mpz_init(ct);
   mpz_init(pt1);
   mpz_init(pt2);

   read_large_int("ciphertext.txt", ct);
   char text[1024];
   size_t count;

   for (int i = 0; i < compromised_keys.size(); i++) {

      mpz_set(n1, intlist[compromised_keys[i].first]);
      mpz_set(n2, intlist[compromised_keys[i].second]); 
      mpz_gcd(p, n1, n2);
     
      break_encryption(pt1, ct, n1, p); 
      break_encryption(pt2, ct, n2, p);

      mpz_export(text, &count, 1, 1, 0, 0, pt1);
      text[count] = '\0';
      print_mp("using public key: ", n1);
      printf("decrytped message is: %s\n", text);

      printf("\n");

      mpz_export(text, &count, 1, 1, 0, 0, pt2);
      text[count] = '\0';
      print_mp("using public key: ", n2);
      printf("decrytped message is: %s\n", text);
   }
}

void break_encryption(mpz_t plaintext, mpz_t ciphertext, mpz_t n, mpz_t p) {

   mpz_t q;

   mpz_init(q);

   mpz_divexact(q, n, p);
   TextbookRSA rsa(p, q);
   rsa.decrypt_message(plaintext, ciphertext);
}

void read_large_int(char* filename, mpz_t i) {

   FILE* file = fopen(filename, "r");

   mpz_inp_str(i, file, 10);

   fclose(file);
}

