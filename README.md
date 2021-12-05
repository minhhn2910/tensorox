# Tensorox
Approximating programs with CUDA tensor core instructions.

This repository shows how to use tensor instructions (e.g. wmma::mma_sync) to approximate part of CUDA programs. 
### Compability and requirement
  * This project requires CUDA install and Nvidia GPU with compute capability at least 7.0 for tensorcore to work. Older architectures will simply result in compilation error. 
  
  * Python 3 with sklearn are required. A few benchmarks may require installing additional libraries, which will be noted at the README file inside the respective subdirectory.
  
  * Tested environment: Ubuntu 18.04, Python 3, Cuda 10.0, Nvidia Volta GPU
  
### Code Hierachy:  

* The training script using sklearn is provided at : training_scripts/ folder. There are two main scripts: the training script and the testing script (i.e. wmma_test). The training script will read config parameter on where the input are output located as well as the layers configuration. The config file and trained weights are available at a separate folder. The sample test script and test data are also provided. It will test the trained network and print out weights in their arranged order that is ready for deployment to CUDA kernels.

* The trained weights provided at training_scripts/weights folder

* The trained weights are also written as predefined constants into the approximated kernels in their respective folder.

* The approximated kernels and the original versions are available in benchmarks/ folder.

* Generally, each benchmark will have an original version in 1 folder (e.g. inversek2j/ ), the tensor version (100% approximated) is in the second folder (e.g. inversek2j_approx/ ). The auto tuning version to adjust speed/accuracy tradeoff by mixing both tensor version and float version are available at the third folder for each benchmark (e.g. inversek2j_approx_mixed).

* Helper functions and tools to rewrite your programs are available at benchmarks/scripts/ folder.

Please read the README file in each subfolder for further instruction on how to run each type of benchmarks because they belong to different benchmark suites.

### License: 
 The code written by me is free for use with any purposes follow MIT license. However, the subdirectories contain benchmarks from various source and have their own license.
