#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>

#include "mp.h"
#include "mp_cuda.h"
#include "pairwise_gcd.h"
#include "gmp_mp_helper.h"

#define BLOCK_WIDTH 32
#define GRID_WIDTH 10

#define RES_SIZE (GRID_WIDTH*GRID_WIDTH*BLOCK_WIDTH*BLOCK_WIDTH)
#define RES_WIDTH (GRID_WIDTH*BLOCK_WIDTH)

#define MAX_SIZE 200000
#define FILE_MAX 4096

void verify_arguments(int argc, char** argv, char* filename);

static void HandleError( cudaError_t err, const char *file, int line){
    if (err != cudaSuccess) {
       printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line); 
       exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



__global__ void mp_kernel(result_keys *res, mp_int* keys, int num_keys, int res_width, int idx_x, int idx_y) {
   
   
   int tid_y = blockIdx.y * blockDim.y + threadIdx.y; 
   int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
   int row = tid_y + idx_y;
   int col = tid_x + idx_x;  
   mp_int cf; 
   cu_mp_init(&cf); 

   mp_int one; 
   cu_mp_init(&one);
   one.idx[0] = 1; 
    

   

   if(row > col && row < num_keys){ 
      cu_mp_int_gcd(&cf, &keys[row], &keys[col]);
      if(cu_mp_int_gt(&cf, &one)){
         res[tid_y * res_width + tid_x].idx_a = row; 
         res[tid_y * res_width + tid_x].idx_b = col;   
	      
      }
      else{
         res[tid_y * res_width + tid_x].idx_a = 0; 
         res[tid_y * res_width + tid_x].idx_b = 0;  
	   }
   }
   __syncthreads();
   


}


void cuda_call(int num_keys, mp_int *keys, result_keys *res){


   int i,j;
   int num_calls;
   
   

   mp_int *keys_d; 
   result_keys  *res_d;
   
   
   num_calls = num_keys/RES_WIDTH;
   if(num_keys % RES_WIDTH){
      num_calls++;
   }

   dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH);
   dim3 dimGrid(GRID_WIDTH,GRID_WIDTH);
   HANDLE_ERROR(cudaMalloc((void **) &keys_d, sizeof(mp_int)*num_keys));
   HANDLE_ERROR(cudaMemcpy(keys_d, keys, sizeof(mp_int)*num_keys, cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMalloc((void **) &res_d, sizeof(result_keys)*RES_SIZE));

  printf("in cuda call num calls %d\n", num_calls);  
   for(i=0; i<num_calls; i+= RES_WIDTH) {
      
      for(j=0; j<num_calls; j+= RES_WIDTH) {
       
         if(i >= j) {
      
            mp_kernel<<<dimGrid,dimBlock>>>(res_d, keys_d, num_keys, RES_WIDTH, i, j);
            
            HANDLE_ERROR(cudaMemcpy(res, res_d, sizeof(result_keys)*RES_SIZE, cudaMemcpyDeviceToHost));
            printf("returned\n"); 
         }
      }
   }
   

}
  

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


int main(int argc, char** argv) {

   
   result_keys *comp_key_idxs;
   int size=0;
   char filename[FILE_MAX];
   mp_int *intlist = (mp_int *) malloc(sizeof(mp_int)*MAX_SIZE);

   verify_arguments(argc, argv, filename);

   size = parse_largeint_file(filename, intlist, MAX_SIZE, 0);
   
   printf("read %d keys from file %s\n", size, filename);
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

