#include "mp_cuda.h"

int parse_largeint_file(char* filename, mp_int* intlist, int max_size, int verbose) {

   FILE* file = fopen(filename, "r");

   mpz_t tmp;
   mpz_init(tmp);

   int i = 0;

   while (mpz_inp_str(tmp, file, 10) > 0 && i < max_size) {

      if (verbose) {
         printf("n%d = ", i+1);
         mpz_out_str(stdout, 10, tmp);
         printf("\n\n");
      }

      mp_init(&intlist[i]);
      mp_import_mpz(&intlist[i], tmp);

      i++;
   }

   fclose(file);

   return i;
}

void print_mpz_hex(char* description, mpz_t num) {

   printf("%s", description);
   mpz_out_str(stdout, 16, num);
   printf("\n");
}

void print_mp_hex(char* description, mp_int* num) {

   printf("%s", description);
   mp_int_print_hex(num);
   printf("\n");
}

void mp_import_mpz(mp_int* dest, mpz_t source) {

   size_t count;

   mpz_export(&dest->idx[0], &count, -1, sizeof(unsigned int), 0, 0, source);
}

void mp_export_mpz(mpz_t dest, mp_int* source) {

   mpz_import(dest, NUM_WORDS, -1, sizeof(unsigned int), 0, 0, &source->idx[0]);
}



__host__ __device__ void mp_init(mp_int* res) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      res->idx[i] = 0;
   }
}

__device__ void mp_int_copy(mp_int* dest, mp_int* source) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      dest->idx[i] = source->idx[i];
   }
}

__host__ __device__ void mp_int_print_hex(mp_int* num) {

   int i;
   int print_zero = 0;

   for (i = NUM_WORDS-1; i >= 0; i--) {
      if (num->idx[i] || print_zero) {
         printf("%.8x", num->idx[i]);
         print_zero = 1;
      }
   }
}

__device__ void mp_int_gcd(mp_int* res, mp_int* lhs, mp_int* rhs) {

   int i;
   int a_even, b_even;
   int done = FALSE;
   int num_shifts = 0;
   mp_int a, b;

   mp_int_copy(&a, lhs);
   mp_int_copy(&b, rhs);

   mp_init(res);
   res->idx[NUM_WORDS-1] = 1;

  

   while (!done) {

      //printf("on iteration: %d\n", iteration++);
      //printf("last words are: %u, %u\n", a.idx[0], b.idx[0]);


      if (mp_int_is_zero(&a) || mp_int_is_zero(&b))
         break;

      a_even = mp_int_is_even(&a);
      b_even = mp_int_is_even(&b);

      if (a_even && b_even) {
         num_shifts++;
         mp_int_shift_right(&a);
         mp_int_shift_right(&b);
      }
      else if (a_even && !b_even) {
         mp_int_shift_right(&a);
      }
      else if (!a_even && b_even) {
         mp_int_shift_right(&b);
      }
      else { // both are odd

         if (mp_int_equal(&a, &b)) {
            mp_int_copy(res, &a);
            done = TRUE;
         }
         else if (mp_int_lt(&a, &b)) {
            mp_int_sub(&b, &b, &a);
            mp_int_shift_right(&b);
         }
         else {
            mp_int_sub(&a, &a, &b);
            mp_int_shift_right(&a);
         }
      }
   }

   for (i = 0; i < num_shifts; i++) {
      mp_int_shift_left(res);
   }
}

__device__ void mp_int_sub(mp_int* res, mp_int* a, mp_int* b) {

   int i, j; 
   mp_int lhs, rhs;


   mp_int_copy(&lhs, a);
   mp_int_copy(&rhs, b);

   //printf("NUM_WORDS is: %d\n", NUM_WORDS);

   for (i = NUM_WORDS - 1; i >= 0; i--) {

      //printf("idx: %d, lhs: %u, rhs %u\n", i, a->idx[i], b->idx[i]);

      if (lhs.idx[i] >= rhs.idx[i]) {
         res->idx[i] = lhs.idx[i] - rhs.idx[i];
      }
      else { // need to borrow
         j = i + 1;
         //printf("start borrow idx: %d\n", j);
         while (res->idx[j] == 0) {
            res->idx[j] = UINT_MAX;
            j++;
         }
         //printf("borrowing from index %d\n", j);
         res->idx[j] -= 1;

         res->idx[i] = UINT_MAX - rhs.idx[i];
         res->idx[i] += lhs.idx[i] + 1;
      }
   }
}

__device__ void mp_int_shift_left(mp_int* res) {

   int i;

   for (i = NUM_WORDS - 1; i >= 0; i--) {

      res->idx[i] = res->idx[i] << 1;
      
      if (i > 0 && res->idx[i-1] & MOST_SIG_BIT) {
         res->idx[i] += LEAST_SIG_BIT;
      }
   }
}

__device__ void mp_int_shift_right(mp_int* res) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {

      res->idx[i] = res->idx[i] >> 1;
      
      if (i < NUM_WORDS - 1 && res->idx[i+1] & LEAST_SIG_BIT) {
         res->idx[i] += MOST_SIG_BIT;
      }
   }
}

__device__ int mp_int_gt(mp_int* lhs, mp_int* rhs) {

   int i;

   for (i = NUM_WORDS - 1; i >= 0; i--) {
      if (lhs->idx[i] > rhs->idx[i]) {
         return TRUE;
      }
      else if (rhs->idx[i] > lhs->idx[i]) {
         return FALSE;
      }
   }

   return FALSE;
}

__device__ int mp_int_gte(mp_int* lhs, mp_int* rhs) {

   return (!mp_int_gt(rhs, lhs));
}

__device__ int mp_int_lt(mp_int* lhs, mp_int* rhs) {

   return mp_int_gt(rhs, lhs);
}

__device__ int mp_int_lte(mp_int* lhs, mp_int* rhs) {

   return (!mp_int_gt(lhs, rhs));
}

__device__ int mp_int_is_odd(mp_int* num) {

   return (num->idx[0] & 1);
}

__device__ int mp_int_is_even(mp_int* num) {

   return (!mp_int_is_odd(num));
}

__device__ int mp_int_is_zero(mp_int* num) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      if (num->idx[i]) {
         return FALSE;
      }
   }

   return TRUE;
}

__device__ int mp_int_equal(mp_int* a, mp_int* b) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      if (a->idx[i] != b->idx[i])
         return FALSE;
   }

   return TRUE;
}

__global__ void mp_kernel(result_keys *res, mp_int* keys, int num_keys, int res_width, int idx_x, int idx_y) {
   
   
   int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
   int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
   int row = tid_y + idx_y;
   int col = tid_x + idx_x;  
   mp_int cf; 
   mp_init(&cf); 

   mp_int one; 
   mp_init(&one);
   one.idx[0] = 1; 
    

   

   if(tid_x > tid_y && row < num_keys && col < num_keys){ 
      mp_int_gcd(&cf, &keys[col], &keys[row]);
      if(mp_int_gt(&cf, &one)){
         res[tid_y * res_width + tid_x].idx_a = col; 
         res[tid_y * res_width + tid_x].idx_b = row;   
	      
      }
      else{
         res[tid_y * res_width + tid_x].idx_a = 0; 
         res[tid_y * res_width + tid_x].idx_b = 0;  
	   }
   }
   else{
         res[tid_y * res_width + tid_x].idx_a = 0; 
         res[tid_y * res_width + tid_x].idx_b = 0;  
	}
   __syncthreads();
   


}


void cuda_call(int num_keys, mp_int *keys, result_keys *res){


   int i,j, k, idx_x,idx_y;
   int num_calls;
   
   

   mp_int *keys_d; 
   result_keys  *res_d;
   
   mpz_t tmp;
   mpz_init(tmp);
   
   
   
   
   num_calls = num_keys/RES_WIDTH;
   if(num_keys % RES_WIDTH){
      num_calls++;
   }
   
            

   dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH);
   dim3 dimGrid(GRID_WIDTH,GRID_WIDTH);
   HANDLE_ERROR(cudaMalloc((void **) &keys_d, sizeof(mp_int)*num_keys));
   HANDLE_ERROR(cudaMemcpy(keys_d, keys, sizeof(mp_int)*num_keys, cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMalloc((void **) &res_d, sizeof(result_keys)*RES_SIZE));

  //printf("in cuda call num calls %d\n", num_calls);  
  //printf("result width = %d\n", RES_WIDTH);
  idx_x = 0;  
  //printf("result size = %d num_keys %d \n", RES_SIZE, num_keys);
   for(i=0; i<num_calls; i++) {
      idx_y =0; 
      
      for(j=0; j<num_calls; j++) {
       
         if(idx_x <= idx_y) {
         
      	  // printf("idx_x %d idx_y %d\n", idx_x, idx_y); 
            mp_kernel<<<dimGrid,dimBlock>>>(res_d, keys_d, num_keys, RES_WIDTH, idx_x, idx_y);
            
            HANDLE_ERROR(cudaMemcpy(res, res_d, sizeof(result_keys)*RES_SIZE, cudaMemcpyDeviceToHost));
           
            
            for(k=0; k< RES_SIZE; k++){
                if(res[k].idx_a != 0 || res[k].idx_b != 0){
            	   //printf(" at k = %d a %u, b %u \n", k, res[k].idx_a, res[k].idx_b);ß 
            	   mp_export_mpz(tmp, &keys[res[k].idx_b]);
                  mpz_out_str(stdout, 10, tmp);
                  printf("\n"); 
                  mp_export_mpz(tmp, &keys[res[k].idx_a]);
                  mpz_out_str(stdout, 10, tmp);
                  printf("\n"); 
            	 }
            } 
           // printf("returned\n"); 
            
         }
         
         idx_y += RES_WIDTH;
      }
      idx_x += RES_WIDTH;
      
   }
   

}
  



int main(int argc, char** argv) {

   
   result_keys *comp_key_idxs;
   int size=0;
   char filename[FILE_MAX];
   mp_int *intlist = (mp_int *) calloc(sizeof(mp_int)*MAX_SIZE, 1);

   verify_arguments(argc, argv, filename);

   size = parse_largeint_file(filename, intlist, MAX_SIZE, 0);
   
   comp_key_idxs = (result_keys*) calloc(RES_SIZE * sizeof(result_keys), 1); 
  // compromizable_pairs = pairwise_gcd(intlist, size, &comp_key_idxs, 1);
   cuda_call(size, intlist, comp_key_idxs); 
   free(intlist);

   return 0;
}

void verify_arguments(int argc, char** argv, char* filename) {

   if (argc < 2) {
      fprintf(stderr, "invalid argument list, requires: %s filename\n", argv[0]);
      exit(0);
   }
   else {
      strncpy(filename, argv[1], FILE_MAX - 1);
   }
}

