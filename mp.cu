#include "mp.h"
#include <stdio.h>


static void HandleError( cudaError_t err, const char *file, int line){
    if (err != cudaSuccess) {
       printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line); 
       exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



__device__ void mp_init(mp_int* res) {

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

__device__ void mp_int_print_hex(mp_int* num) {

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

   int iteration = 0;

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

   int i, j, borrow, current; 
   mp_int lhs, rhs;

   borrow = 0;

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

int mp_int_gte(mp_int* lhs, mp_int* rhs) {

   return (!mp_int_gt(rhs, lhs));
}

int mp_int_lt(mp_int* lhs, mp_int* rhs) {

   return mp_int_gt(rhs, lhs);
}

int mp_int_lte(mp_int* lhs, mp_int* rhs) {

   return (!mp_int_gt(lhs, rhs));
}

int mp_int_is_odd(mp_int* num) {

   return (num->idx[0] & 1);
}

int mp_int_is_even(mp_int* num) {

   return (!mp_int_is_odd(num));
}

int mp_int_is_zero(mp_int* num) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      if (num->idx[i]) {
         return FALSE;
      }
   }

   return TRUE;
}

int mp_int_equal(mp_int* a, mp_int* b) {

   int i;

   for (i = 0; i < NUM_WORDS; i++) {
      if (a->idx[i] != b->idx[i])
         return FALSE;
   }

   return TRUE;
}

__global__ void mp_kernel(mp_int* res, mp_int* a, int idx_x, int idx_y) {
   
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   
   if(threadIdx.x > threadIdx.y){
   //gdc calculation using idx_x and idx_y

   
   }
   


}


void cuda_call(int num_keys, mp_int *keys, mp_int *res){


   int idx_x, idx_y, i,j;
   int num_blocks;
   
   

   mp_int keys_d; 
   mp_int res_d;
   //BLOCK_WIDTH is going to also change depending on how many keys
   num_blocks = num_keys/BLOCK_WIDTH;
   if(num_keys % BLOCK_WIDTH)
      num_blocks++;

   dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH);
   //dim3 dimGrid(numBlocks,numBlocks);
   HANDLE_ERROR(cudaMalloc((void **) &keys_d.idx, sizeof(mp_int)*num_keys));
   HANDLE_ERROR(cudaMemcpy(keys_d.idx, keys->idx, sizeof(mp_int)*num_keys, cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMalloc((void **) &res_d.idx, sizeof(mp_int)*BLOCK_WIDTH*BLOCK_WIDTH));



   idx_x=0; 
   idx_y=0;
   for(i=0; i<num_blocks; i+=BLOCK_WIDTH) {
      
      for(j=0; j<num_blocks; j+=BLOCK_WIDTH) {
       
         if(i>=j) {
      
            multKernel<<<1,dimBlock>>>(res_d, keys_d, i, j);
            HANDLE_ERROR(cudaMemcpy(res->idx, res_d.idx, sizeof(mp_int)*num_keys, cudaMemcpyDeviceToHost));
            //need another for loop to take out the values that are not 1
         }
      }
   }

         


      















   return 0;
}

