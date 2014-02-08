#include <stdio.h>
#include <gmp.h>
#include <time.h>

int gcd(int a , int b);

int main() {

   srand(time(0));

   int a = rand();
   int b = rand();

   printf("my gcd function\n");
   printf("gcd of %d and %d is: %d\n\n", a, b, gcd(a, b));

   mpz_t mpz_a;
   mpz_t mpz_b;
   mpz_t mpz_result;

   mpz_init(mpz_a);
   mpz_init(mpz_b);
   mpz_init(mpz_result);

   mpz_set_ui(mpz_a, a);   
   mpz_set_ui(mpz_b, b);   

   mpz_gcd(mpz_result, mpz_a, mpz_b);

   printf("gmp gcd function\n");
   printf("gcd of ");
   mpz_out_str(stdout, 10, mpz_a);
   printf(" and ");
   mpz_out_str(stdout, 10, mpz_b);
   printf(" is: ");
   mpz_out_str(stdout, 10, mpz_result);
   printf("\n");

   return 0;
}

int gcd(int a, int b) {

   int a_even = (a%2 == 0);
   int b_even = (b%2 == 0);
  
   if (a == 0 || b == 0)
      return 0;
 
   if (a_even && b_even) {
      return 2*gcd(a/2, b/2);
   }
   else if (a_even && !b_even) {
      return gcd(a/2, b);
   }
   else if (!a_even && b_even) {
      return gcd(a, b/2);
   }
   else { // both are odd

      if (a == b)
         return a;
      else if (a < b)
         return gcd((b-a)/2, a);
      else
         return gcd((a-b)/2, b);
   }
}
