#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string>
#include "TextbookRSA.h"
#include "gmp_helper.h"

using namespace std;

int main(void) {

   TextbookRSA t;

   mpz_t pt, ct, pt2;
   mpz_init(pt);
   mpz_init(ct);
   mpz_init(pt2);

   mpz_set_ui(pt, 65);

   t.encrypt_message(pt, ct);
   t.decrypt_message(pt2, ct);

   print_mp("original message: ", pt);
   print_mp("ciphertext: ", ct);
   print_mp("decrypted message: ", pt2);

   return 0;
}
