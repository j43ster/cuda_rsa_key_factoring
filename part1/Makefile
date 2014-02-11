all: part1 part2 part3 part4 part5

part1:
	gcc -o part1 part1.c -I/home/clupo/gmp/include -L/home/clupo/gmp/lib -lgmp

part2:
	gcc -o part2 part2.c -I/home/clupo/gmp/include -L/home/clupo/gmp/lib -lgmp 

part3:
	g++ -m64 -o part3 part3.cpp pairwise_gcd.cpp -I/home/clupo/gmp/include /home/clupo/gmp/lib/libgmp.a 

part4:
	g++ -o part4 part4.cpp gmp_helper.cpp TextbookRSA.cpp -I/home/clupo/gmp/include -L/home/clupo/gmp/lib -lgmp 

part5:
	g++ -o part5 part5.cpp gmp_helper.cpp TextbookRSA.cpp pairwise_gcd.cpp -I/home/clupo/gmp/include -L/home/clupo/gmp/lib -lgmp 

format_converter:
	g++ -o format_converter format_converter.cpp gmp_helper.cpp TextbookRSA.cpp pairwise_gcd.cpp -I/home/clupo/gmp/include /home/clupo/gmp/lib/libgmp.a

clean:
	rm -f part1 part2 part3 part4 part5 format_converter
