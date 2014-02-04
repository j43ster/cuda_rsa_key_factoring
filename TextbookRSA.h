#ifndef TEXTBOOKRSA_H
#define TEXTBOOKRSA_H

#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string>
#include "gmp_helper.h"

class TextbookRSA {

   public:
 
      static const unsigned long int E = 65537;
      static const int NUM_BITS = 1024;

      TextbookRSA();
      TextbookRSA(mpz_t p, mpz_t q);

      void encrypt_message(mpz_t plaintext, mpz_t ciphertext);
      void decrypt_message(mpz_t plaintext, mpz_t ciphertext);

   private:

      mpz_t m_n; // public key modulus
      mpz_t m_e; // public key exponent
      mpz_t m_d; // private key

      void generate_keys();
      void generate_keys(mpz_t p, mpz_t q);

};

#endif
