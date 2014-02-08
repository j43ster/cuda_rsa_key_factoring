#!/bin/bash

#./test.sh <program location> <keyfile location> <outfile destination path> <verification file path>

FILES=$2/*

for file in $FILES
do
   NUM=${file##*/}
   echo "      ./gcd_cpu $file > $3/$NUM.out"
   time $1 $file > "$3/$NUM.out"
done
