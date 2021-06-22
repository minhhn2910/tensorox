#!/usr/bin/python

import sys
import math


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printUsage():
	print "Usage: python qos.py <original file> <nn file>"
	exit(1)
pass;

if(len(sys.argv) != 3):
	printUsage()

origFilename 	= sys.argv[1]
nnFilename		= sys.argv[2]

origLines 		= open(origFilename).readlines()
nnLines			= open(nnFilename).readlines()

mismatch        = 0

for i in range(len(origLines)):
    origLine 	= int(origLines[i].rstrip())
    nnLine 		= int(nnLines[i].rstrip())
    if(origLine != nnLine):
        mismatch += 1

print bcolors.FAIL	+ "*** Error: %1.8f%%" % ((mismatch/float(len(origLines)))*100) + bcolors.ENDC
