#Use this script to verify the correctness and completeness of your key-breaking software
#The script expects as input two files. The first is a list of known bad keys (we've provided 
#you some for the short key files); the second is your candidate list of broken moduli (e) and 
#corresponding private key (d). The format of the first file is a string, decimal representation 
#of an RSA moduli, one per line. The format of the second file is n:d, where n and d are string, 
#decimal representation of an RSA moduli and its corresponing private key d, delimited by a colon, 
#also one per line.

import sys, random


#Open the known badkeys file
bk = open(sys.argv[1])
#Open facroted keys file
f = open(sys.argv[2])

e = 65537

badkeys = []
keys = {}
#Get a list of the known bad keys
for n in bk:
	badkeys.append(n[:-1])

#Parse the keys file as modulus:d, and throw them into a dictory
for l in f:
	nd = l.split(":")
	if nd[0] not in keys:
		keys[nd[0]] = nd[1][:-1]

#Check to see if we've found all the keys
missing = 0
for n in badkeys:
	if n not in keys:
		missing += 1
print "Missing", missing, "bad keys."

#Check for correctness
incorrect = 0
for n,d in keys.iteritems():
	pt = random.randint(1,int(n))
	c = pow(pt,e,int(n))
	m = pow(c,int(d),int(n))
	if pt != m:
		incorrect += 1

print "Cracked", len(keys) - incorrect, "of", len(keys), "keys."
