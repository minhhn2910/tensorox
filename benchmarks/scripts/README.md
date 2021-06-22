This is a POC script for rewriting CUDA code using trained weights. Requirements: pickle(protocol=3) of python3 and numpy
usage python3 rewrite.py config.json

### Code annotation:

To hint the converter for where to put the array constant and helper function declaration //tensorox_define

Put this line at the desired place, preferably at the beginning of the file after #includes


The approximated code region must be wrapped around the two hints : //tensorox_input and //tensorox_output
following //tensorox_input and //tensorox_output are the list of variable separated by colons [:]. One example is as below:

... some other code ...

//tensorox_input:var_1:var_2:var_3

[the code to be approximated]

//tensorox_output:out_1:out_2

Please refer to the sample.cu for actual annotation.

### Config file:

JSON format, the below is the meaning of the parameters:

* "layer_dimension":[4,8,8,6], # network architecture, currently not used, can be skipped.

* "weight_file": "invk2j.weights", #weight file saved from training scripts.

* "bias_file" : "invk2j.bias", #bias file saved from training scripts

* "source_file" : "sample.cu", #source_file with at least annotation for tensorox_input and tensorox_output, see sample.cu for how to annotate

* "destination_file" : "sample_approx.cu", #file to write the approximated tensor version of the source file.

* "means_in" : [-0.00047858, 1.4443, -0.0013808, 1.445], #input scaling parameters, means, can leave as [] empty array or [0,0,0,....] if not used.

* "scales_in" : [0.8009, 1.4939, 0.8002, 1.4952], #input scaling parameters, scales, can leave as [] empty array or [1,1,1,....] if not used.

* "scales_out" : [100,100,100,100,100,100] #output scaling, to be multiplied with output after MLP, can leave as [] empty array or [1,1,1,....] if not used.
