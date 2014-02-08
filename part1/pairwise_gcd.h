#ifndef PAIRWISE_GCD_H
#define PAIRWISE_GCD_H

#include <stdio.h>
#include <gmp.h>
#include <time.h>
#include <string>
#include <utility>
#include <vector>

using namespace std;

int parse_largeint_file(string filename, mpz_t* intlist, int max_size, bool verbose);

// returns a vector of pairs of indexes into intlist of RSA public keys
// that share a factor
vector<pair<int, int> > pairwise_gcd(mpz_t* intlist, int length, bool verbose);

#endif
