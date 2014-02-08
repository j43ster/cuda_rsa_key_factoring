#include <stdio.h>
#include <gmp.h>
#include <time.h>

void read_write_large_ints();
void basic_gmp();
void write_large_int(char* filename, mpz_t i);
void read_large_int(char* filename, mpz_t i);
void add_subtract_multiply(mpz_t a, mpz_t b);
void mod_exp(mpz_t a, mpz_t b, mpz_t c);

int main() {

   mpz_t a;
   mpz_t b;
   mpz_t c;

   mpz_init(a);
   mpz_init(b);
   mpz_init(c);

   mp_bitcnt_t num_bits = 512;

   gmp_randstate_t randstate;
   gmp_randinit_mt(randstate);   
   gmp_randseed_ui(randstate, time(0));

   mpz_urandomb(a, randstate, num_bits);
   mpz_urandomb(b, randstate, num_bits);
   mpz_urandomb(c, randstate, num_bits);

   // parts 1-3
   basic_gmp();

   fprintf(stdout, "\n");

   // part 4
   read_write_large_ints();

   // part 5-6
   add_subtract_multiply(a, b);

   // part 7
   mod_exp(a, b, c);

   return 0;
}

void mod_exp(mpz_t a, mpz_t b, mpz_t c) {

   mpz_t res;
   mpz_init(res);

   fprintf(stdout, "a: ");
   mpz_out_str(stdout, 10, a);
   fprintf(stdout, "\n\n");
    
   fprintf(stdout, "b: ");
   mpz_out_str(stdout, 10, b);
   fprintf(stdout, "\n\n");

   fprintf(stdout, "c: ");
   mpz_out_str(stdout, 10, c);
   fprintf(stdout, "\n\n");

   mpz_powm(res, a, b, c);
    
   fprintf(stdout, "(a^b)modc: ");
   mpz_out_str(stdout, 10, res);
   fprintf(stdout, "\n\n");
}

void add_subtract_multiply(mpz_t a, mpz_t b) {

   FILE* outfile = stdout;

   mpz_t c;
   mpz_init(c);

   fprintf(stdout, "a: ");
   mpz_out_str(stdout, 10, a);
   fprintf(stdout, "\n\n");
    
   fprintf(stdout, "b: ");
   mpz_out_str(stdout, 10, b);
   fprintf(stdout, "\n\n");

   mpz_add(c, a, b);
   
   fprintf(stdout, "a+b: ");
   mpz_out_str(stdout, 10, c);
   fprintf(stdout, "\n\n");

   mpz_sub(c, a, b);
   
   fprintf(stdout, "a-b: ");
   mpz_out_str(stdout, 10, c);
   fprintf(stdout, "\n\n");

   mpz_mul(c, a, b);
   
   fprintf(stdout, "a*b: ");
   mpz_out_str(stdout, 10, c);
   fprintf(stdout, "\n\n");
}

void basic_gmp() {

   FILE* outfile = stdout;

   mpz_t i;
   mpz_init(i);

   mp_bitcnt_t num_bits = 512;

   gmp_randstate_t randstate;
   gmp_randinit_mt(randstate);   

   // 1. generate psuedo random number
   gmp_randseed_ui(randstate, time(0));
   mpz_urandomb(i, randstate, num_bits);

   // 3. write large integer to stdout
   mpz_out_str(outfile, 10, i);
   fprintf(outfile, "\n");

   // 2. check to see if is probabilistically prime
   int res = mpz_probab_prime_p(i, 0.25); 

   if (res == 2)
      fprintf(outfile, "Is definitely a prime\n");
   else if (res == 1)
      fprintf(outfile, "Is probably a prime\n");
   else
      fprintf(outfile, "Is definitely not a prime\n");
}

void read_large_int(char* filename, mpz_t i) {

   FILE* file = fopen(filename, "r");

   mpz_inp_str(i, file, 10);

   fclose(file);
}

void write_large_int(char* filename, mpz_t i) {

   FILE* file = fopen(filename, "w");

   mpz_out_str(file, 10, i);

   fclose(file);
}

// read and write larger integers from disk
// reads a large number from file "large_integer.txt"
// uses the number as a random seed and writes a new random
// back to the file
void read_write_large_ints() {

   mpz_t i;
   mpz_init(i);
   read_large_int("largeint.txt", i);

   gmp_randstate_t randstate;
   gmp_randinit_mt(randstate);
   gmp_randseed(randstate, i);

   fprintf(stdout, "read from file: ");
   mpz_out_str(stdout, 10, i);
   fprintf(stdout, "\n\n");

   mpz_urandomb(i, randstate, 512);
   
   fprintf(stdout, "writing to file: ");
   mpz_out_str(stdout, 10, i);
   fprintf(stdout, "\n\n");

   write_large_int("largeint.txt", i);
}
