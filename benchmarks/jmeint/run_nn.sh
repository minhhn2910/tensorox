#!/bin/bash

# Regular Colors
Black='\e[0;30m'        # Black
Red='\e[0;31m'          # Red
Green='\e[0;32m'        # Green
Yellow='\e[0;33m'       # Yellow
Blue='\e[0;34m'         # Blue
Purple='\e[0;35m'       # Purple
Cyan='\e[0;36m'         # Cyan
White='\e[0;37m'        # White


echo -e "${Green} CUDA jmeint Starting... ${White}"

if [ ! -d ./train.data/output/kernel.data ]; then
	mkdir ./train.data/output/kernel.data
fi

for f in test.data/input/*.data
do
	filename=$(basename "$f")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo -e "-------------------------------------------------------"
	echo -e "${Green} Input Locations File:  $f"
	echo -e "${Green} Intersection File: ./test.data/output/${filename}_intersect.data ${White}"
	echo -e "${Green} Intersection File: ./test.data/output/${filename}_intersect_nn.data ${White}"
	echo -e "-------------------------------------------------------"
	./bin/jmeint.out $f ./test.data/output/${filename}_intersect.data
	./bin/jmeint_nn.out $f ./test.data/output/${filename}_intersect_nn.data
	python ./script/qos.py ./test.data/output/${filename}_intersect.data ./test.data/output/${filename}_intersect_nn.data
done
