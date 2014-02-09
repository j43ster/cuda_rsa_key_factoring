#ifndef MP_INT_H
#define MP_INT_H

#include <limits.h>

#define MOST_SIG_BIT ((unsigned int)(1 << (sizeof(unsigned int)*8 - 1)))
#define LEAST_SIG_BIT 1

#define NUM_BITS 1024
#define NUM_WORDS (NUM_BITS/(8*sizeof(unsigned int)))

#define TRUE 1
#define FALSE 0 

// least significant word first
typedef struct mp_int {
    unsigned int idx[NUM_WORDS];
} mp_int;

typedef struct result_keys {

   int idx_a;
   int idx_b;

} result_keys;

#endif
