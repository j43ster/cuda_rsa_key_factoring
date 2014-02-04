#include "TextbookRSA.h"

TextbookRSA::TextbookRSA() {

   mpz_init(m_n);
   mpz_init(m_e);
   mpz_init(m_d);

   mpz_set_ui(m_e, E);

   generate_keys();
}

TextbookRSA::TextbookRSA(mpz_t p, mpz_t q) {

   mpz_init(m_n);
   mpz_init(m_e);
   mpz_init(m_d);

   mpz_set_ui(m_e, E);

   generate_keys(p, q);
}

void TextbookRSA::encrypt_message(mpz_t plaintext, mpz_t ciphertext) {

   mpz_powm(ciphertext, plaintext, m_e, m_n);
}

void TextbookRSA::decrypt_message(mpz_t plaintext, mpz_t ciphertext) {

   mpz_powm(plaintext, ciphertext, m_d, m_n);
}

// this method is not safe because it does not check that the 
// p and q provided are legit
// misuse will result in silent failure
// also does not check to ensure that denom is 1
void TextbookRSA::generate_keys(mpz_t p, mpz_t q) {

   mpz_t phi_n, denom;
   mpz_init(phi_n);
   mpz_init(denom);

   // compute n = pq
   mpz_mul(m_n, p, q);

   // compute phi_n = (p-1)(q-1)
   mpz_sub_ui(p, p, 1);
   mpz_sub_ui(q, q, 1);
   mpz_mul(phi_n, p, q);

   // choose e such that 1 < e < phi_n, and gcd(e, phi_n) = 1 (e and phi_n are coprime)
   // e is 65537, just make sure that gcd e and phi_n is 1
   mpz_gcd(denom, phi_n, m_e);

   // d = mpz_invert, d is he private key
   mpz_invert(m_d, m_e, phi_n);
}

void TextbookRSA::generate_keys() {

   mpz_t p, q, phi_n, denom;
   mpz_init(p);
   mpz_init(q);
   mpz_init(phi_n);
   mpz_init(denom);

   do {
      // chooose two prime numbers p, q
      random_prime(p, NUM_BITS);
      random_prime(q, NUM_BITS);

      // compute n = pq
      mpz_mul(m_n, p, q);

      // compute phi_n = (p-1)(q-1)
      mpz_sub_ui(p, p, 1);
      mpz_sub_ui(q, q, 1);
      mpz_mul(phi_n, p, q);

      // choose e such that 1 < e < phi_n, and gcd(e, phi_n) = 1 (e and phi_n are coprime)
      // e is 65537, just make sure that gcd e and phi_n is 1
      mpz_gcd(denom, phi_n, m_e);

   } while (mpz_cmp_ui(denom, 1));

   // d = mpz_invert, d is he private key
   mpz_invert(m_d, m_e, phi_n);
}

