#!/bin/bash

make clean
make

cp ./part1/part3 ./gcd_cpu_gmp

echo
echo now running:
echo "   small and medium key files using the gmp library"
#./test.sh ./gcd_cpu_gmp "./keyfiles/test" ./results/gmp ./compromised_keys
#./test.sh ./gcd_cpu_gmp "./keyfiles/small" ./results/gmp ./compromised_keys
#./test.sh ./gcd_cpu_gmp "./keyfiles/medium" ./results/gmp ./compromised_keys

echo
echo "   small key files with custom mp library"
./test.sh ./gcd_cpu "./keyfiles/test" ./results/cpu ./compromised_keys
./test.sh ./gcd_cpu "./keyfiles/small" ./results/cpu ./compromised_keys

echo "   all key files using gpu and custom mp library"
#./test.sh ./gcd_gpu "./keyfiles/test" ./results/gpu ./compromised_keys
#./test.sh ./gcd_gpu "./keyfiles/small" ./results/gpu ./compromised_keys
#./test.sh ./gcd_gpu "./keyfiles/medium" ./results/gpu ./compromised_keys
