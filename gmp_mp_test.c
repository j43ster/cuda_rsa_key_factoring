#include <assert.h>
#include <stdio.h>
#include <gmp.h>

#include "mp.h"
#include "gmp_mp_helper.h"
#include "gmp_helper.h"

void import_export_test() {

   mpz_t source;
   mpz_t exports;
   mp_int import;

   mpz_init(source);
   mpz_init(exports);
   mp_init(&import);

   mp_bitcnt_t num_bits = 1024;

   gmp_randstate_t randstate;
   gmp_randinit_mt(randstate);
   gmp_randseed_ui(randstate, time(0));

   mpz_urandomb(source, randstate, num_bits);
 
   mp_import_mpz(&import, source);
   mp_export_mpz(exports, &import);

 //  print_mp("", source);
 //  printf("\n");
 //  print_mp("", exports);

   int res = mpz_cmp(source, exports);
//   printf("res is: %d\n", res);

   assert((mpz_cmp(source, exports) == 0));
}

void sub_test() {

   mpz_t result, lhs, rhs, verify;
   mp_int m_result, m_lhs, m_rhs;

   mpz_init(result);
   mpz_init(lhs);
   mpz_init(rhs);
   mpz_init(verify);

   mp_init(&m_result);
   mp_init(&m_lhs);
   mp_init(&m_rhs);

   mp_bitcnt_t num_bits = 1024;

   gmp_randstate_t randstate;
   gmp_randinit_mt(randstate);
   gmp_randseed_ui(randstate, time(0));

   mpz_urandomb(lhs, randstate, num_bits);
   
   do {
      mpz_urandomb(rhs, randstate, num_bits);
   } while (mpz_cmp(lhs, rhs) < 0);

//   mpz_set_ui(lhs, UINT_MAX);
//   mpz_mul_ui(lhs, lhs, 2);
//   mpz_sub_ui(lhs, lhs, 10);

//   mpz_set_ui(rhs, UINT_MAX);

   mp_import_mpz(&m_lhs, lhs);
   mp_import_mpz(&m_rhs, rhs);

   mpz_sub(result, lhs, rhs);
   mp_int_sub(&m_result, &m_lhs, &m_rhs);

   mp_export_mpz(verify, &m_result);

   print_mp("lhs: ", lhs);
   print_mp("rhs: ", rhs);

   print_mp("correct: ", result);
   printf("\n");
   print_mp("   ours: ", verify);

   assert(mpz_cmp(verify, result) == 0);
}

void gcd_test() {

   mpz_t result, a, b, verify;
   mp_int m_result, m_a, m_b;

   mpz_init(result);
   mpz_init(a);
   mpz_init(b);
   mpz_init(verify);

   mp_init(&m_result);
   mp_init(&m_a);
   mp_init(&m_b);

   mp_bitcnt_t num_bits = 1024;

   gmp_randstate_t randstate;
   gmp_randinit_mt(randstate);
   gmp_randseed_ui(randstate, time(0));

   mpz_urandomb(a, randstate, num_bits);
   mpz_urandomb(b, randstate, num_bits);

   mp_import_mpz(&m_a, a);
   mp_import_mpz(&m_b, b);

   mp_int_gcd(&m_result, &m_a, &m_b);
   mpz_gcd(result, a, b);

   mp_export_mpz(verify, &m_result);

   print_mp(" gcd is: ", result);
   print_mp("our gcd: ", verify);
   assert(mpz_cmp(result, verify) == 0);
}

int main() {

   import_export_test();
   sub_test();
   gcd_test();

   return 0;
}
