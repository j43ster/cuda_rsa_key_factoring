#!/bin/bash

make clean
make

cp ./part1/part3 ./gcd_cpu_gmp

echo
echo now running:
echo "   small and medium key files using the gmp library"
./test.sh ./gcd_cpu_gmp "./keyfiles/test" ./out ./compromised_keys
./test.sh ./gcd_cpu_gmp "./keyfiles/small" ./out ./compromised_keys
./test.sh ./gcd_cpu_gmp "./keyfiles/medium" ./out ./compromised_keys

echo
echo "   small key files with custom mp library"
./test.sh ./gcd_cpu "./keyfiles/test" ./out ./compromised_keys
./test.sh ./gcd_cpu "./keyfiles/small" ./out ./compromised_keys

echo "   all key files using gpu and custom mp library"
