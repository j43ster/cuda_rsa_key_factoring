#ifndef PAIRWISE_GCD_H
#define PAIRWISE_GCD_H

#include <stdio.h>
#include <time.h>
#include <gmp.h>

#include "mp.h"
#include "gmp_helper.h"

#define MAX_COMPROMISED_KEYS 200000

typedef struct compromised_keys {

   int idx_a[MAX_COMPROMISED_KEYS];
   int idx_b[MAX_COMPROMISED_KEYS];

} compromised_keys;

// returns number of compromised keys
int pairwise_gcd(mp_int* intlist, int size, compromised_keys* comp_key_idxs, int verbose);

#endif
