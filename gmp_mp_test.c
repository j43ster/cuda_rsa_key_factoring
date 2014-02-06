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

   print_mp("", source);
   printf("\n");
   print_mp("", exports);

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

   mp_import_mpz(&m_lhs, lhs);
   mp_import_mpz(&m_rhs, rhs);

   mpz_sub(result, lhs, rhs);
   mp_int_sub(&m_result, &m_lhs, &m_rhs);

   mp_export_mpz(verify, &m_result);

   print_mp("correct: ", result);
   printf("\n");
   print_mp("   ours: ", verify);

   assert(mpz_cmp(verify, result) == 0);
}

void gcd_test() {

}

int main() {

   import_export_test();
   sub_test();

   return 0;
}
