#!/usr/bin/python

import sys
import math
import numpy as np

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
	print ("Usage: python qos.py <original file> <nn file>")
	exit(1)
pass;


if(len(sys.argv) != 3):
	printUsage()

origFilename 	= sys.argv[1]
nnFilename		= sys.argv[2]

origLines 		= open(origFilename).readlines()
nnLines			= open(nnFilename).readlines()


e = 0.0
absError = 0.0
rmse = 0.0
err_array = []
for i in range(len(origLines)):

    origLine 	= origLines[i].rstrip()
    nnLine 		= nnLines[i].rstrip()

    origPrice 	= float(origLine)
    nnPrice 	= float(nnLine)

    nominator   = abs(origPrice - nnPrice)
    denominator = abs(origPrice)
    if(denominator == 0):
        e = nominator#1.0  This is to eliminate the case where python qos.py ref.txt ref.txt != 0; where ref has results = 0;
    elif(math.isnan(nominator) or (math.isnan(denominator))):
        e = 1.0
    elif ((nominator / denominator > 1)):
        e = 1.0
    else:
        e = nominator / denominator
    rmse += (nominator*nominator)
    absError += e
    err_array.append(e)
pass;
bins = np.arange(0,1.0,0.05)
hist = np.histogram(err_array,bins)
print ("cdf : ")
pdf = hist[0]/float(len(err_array))
cdf = np.cumsum(pdf)
print (list(cdf))
print (bcolors.FAIL	+ "*** Error: %1.8f" % (absError/float(len(origLines))) + bcolors.ENDC)
