// Designed by: Amir Yazdanbakhsh
// Date: March 26th - 2015
// Alternative Computing Technologies Lab.
// Georgia Institute of Technology


#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <cstddef>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

__constant__ half weight_1_half_d[80];
__constant__ half2 weight_2_half_d[8];

__constant__ half bias_1_half_d[8];


__device__ __inline__ half relu( half x){
  return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0) ;
}

// Cuda Libraries
#include <cuda_runtime_api.h>
#include <cuda.h>

//#define MAX_LOOP 5
#define MAX_LOOP 20 //more loops, even number

#define MAX_DIFF 0.15f

using namespace std;

__global__ void nrpol3_kernel(float *A_coeff, float *B_coeff, float *C_coeff, float *D_coeff, float *x0_in,
  half *A_coeff_half, half *B_coeff_half, half *C_coeff_half, half *D_coeff_half, half *x0_in_half
  , float *root, int size, float err_thresh)
{
  //  if (blockIdx.x !=0)return;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
  	half2 bias_2_dev = __floats2half2_rn(2.98344, -10.2018);
     int real_tid =  threadIdx.x;
      int warp_id = real_tid /32;
      int warp_lane = real_tid %32;

      __shared__ half weight_1_shared[128];
      __shared__ half bias_1_shared[256];
      __shared__ half neuron_out[8][512];

      if (real_tid <80){
        weight_1_shared[real_tid] = weight_1_half_d[real_tid];
      }
      else if (real_tid < 128)
  		weight_1_shared[real_tid] = 0.0;

      //simple trick no need if/else, read as col_major later
  	bias_1_shared[real_tid] = bias_1_half_d[warp_id];

      for (int i = 0; i<8 ; i++){
        neuron_out[i][real_tid] = 0.0;
        neuron_out[i][real_tid+256] = 0.0;
      }
      __syncthreads();

  	//	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag_col;
  		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
  		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

  	//	if (tid < nthreads-1){
  			  neuron_out[warp_id][warp_lane+ 0*32] = A_coeff_half[2*tid];
  			  neuron_out[warp_id][warp_lane+ 1*32] = B_coeff_half[2*tid];
  			  neuron_out[warp_id][warp_lane+ 2*32] = C_coeff_half[2*tid];
  			  neuron_out[warp_id][warp_lane+ 3*32] = D_coeff_half[2*tid];
          neuron_out[warp_id][warp_lane+ 4*32] = x0_in_half[2*tid];

          neuron_out[warp_id][warp_lane+ 5*32] = A_coeff_half[2*tid+1];
  			  neuron_out[warp_id][warp_lane+ 6*32] = B_coeff_half[2*tid+1];
  			  neuron_out[warp_id][warp_lane+ 7*32] = C_coeff_half[2*tid+1];
  			  neuron_out[warp_id][warp_lane+ 8*32] = D_coeff_half[2*tid+1];
          neuron_out[warp_id][warp_lane+ 9*32] = x0_in_half[2*tid+1];
  		// }
        __syncthreads();
//        printf("me %d\n", tid);
//        root[tid] = neuron_out[0][tid];
//        return;

    		wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
    		wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared, 8);
    		wmma::load_matrix_sync(c_frag, (const half*)&bias_1_shared, 32, wmma::mem_col_major);

    		wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
    		for (int i = 0; i< c_frag.num_elements; i ++)
    			c_frag.x[i] = relu(c_frag.x[i]);
   		wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);

  		__syncwarp();
  		half2 output1= __float2half2_rn(0.0);
  		for (int i =0 ; i <8 ; i++ )
  			output1 += __half2half2(neuron_out[warp_id][warp_lane+i*32])*weight_2_half_d[i];
  		output1 += bias_2_dev;//__float2half_rn(BIAS2);

    if (tid*2 < size) {
        root[tid*2] = __half2float(output1.x);
  	   root[tid*2+1] = __half2float(output1.y);

    }


}

float weight_1 [80] = {-0.0138049, -0.00319177, -0.0346182, 0.0250096, -4.46268, 0.144932, -4.25475, -0.444511, -0.00976453, -0.0272127, -0.0194992, 0.00366111, -4.5546, 0.0158578, -1.02843, -0.155528, 0.0104555, -0.00735722, 0.0320725, -0.0131664, -0.562335, 0.000142681, -0.0903253, 0.0125778, 0.00187236, 0.0341705, 0.00428708, 0.011335, -0.1538, 0.108896, -0.0832994, -0.208423, 0.0187008, 0.0204824, 0.00615459, -0.00973818, 0.0601625, 0.00707444, 0.0097333, 0.0691306, -4.44395, 0.150768, -3.95769, 2.2455, -0.00633222, -0.949656, -0.00349646, -0.533252, -2.83163, 0.0806728, -4.17467, 0.485617, 0.00360298, -0.176358, 0.00777581, -0.0964101, -0.22236, 2.13235, -0.654349, 0.0346239, 0.0545703, 1.99659, -0.0111349, 1.20922, -0.138669, 0.0380622, -0.187684, 0.149302, -0.0218722, -0.116879, 0.0265327, -0.055342, 0.0101457, 0.764969, 0.023873, 0.0468976, 0.0486227, 0.710096, -0.000650336, 0.430674};
float bias_1 [8] = {-11.3375, 3.98549, -13.2682, 2.6579, -14.3034, 5.68077, -7.94012, 3.55371};
float weight_2 [16] = {0.046989, 36.3615, -0.0654739, -4.15659, -0.0515215, 43.6671, -0.00569459, 1.86478, 45.3918, -0.0508928, -0.749983, 3.57068, 19.7714, -0.044111, 1.30729, 1.31848};
float bias_2[2] = {2.98344, -10.2018};
half weight_1_half[80], bias_1_half[8], bias_2_half[2];
half2 weight_2_half[8];
void prepare_half_prec_weights(){
		for (int i =0;i < 80; i++)
			weight_1_half[i] = __float2half_rn(weight_1[i]);
		for (int i = 0; i< 8 ;i ++){
			weight_2_half[i] = __floats2half2_rn(weight_2[2*i],weight_2[2*i+1]);
			bias_1_half[i] = __float2half_rn(bias_1[i]);
		}
		for (int i = 0; i<2 ;i ++){

			bias_2_half[i] = __float2half_rn(bias_2[i]);
		}


}


int main(int argc, char* argv[])
{
	if(argc != 4)
	{
		std::cerr << "Usage: ./nrpoly3.out <input file coefficients> <output file> <error threshold>" << std::endl;
		exit(EXIT_FAILURE);
	}
	prepare_half_prec_weights();
	float* A_coeff;
	float* B_coeff;
	float* C_coeff;
	float* D_coeff;
	float* x0;

  half* A_coeff_half;
	half* B_coeff_half;
	half* C_coeff_half;
	half* D_coeff_half;
	half* x0_half;

	float* root;

	cudaError_t cudaStatus;

	int data_size = 0;

	// process the files
	ifstream coeff_in_file (argv[1]);
	ofstream root_out_file (argv[2]);
	float err_thresh = atof(argv[3]);


	if(coeff_in_file.is_open())
	{
		coeff_in_file >> data_size;
		std::cout << "# Data Size = " << data_size << std::endl;
	}

	// allocate the memory
	A_coeff = new (nothrow) float[data_size];
  A_coeff_half = new (nothrow) half[data_size];
	if(A_coeff == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	B_coeff = new (nothrow) float[data_size];
  B_coeff_half = new (nothrow) half[data_size];
	if(B_coeff == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	C_coeff = new (nothrow) float[data_size];
  C_coeff_half = new (nothrow) half[data_size];
	if(C_coeff == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	D_coeff = new (nothrow) float[data_size];
  D_coeff_half = new (nothrow) half[data_size];
	if(D_coeff == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	x0 = new (nothrow) float[data_size];
  x0_half = new (nothrow) half[data_size];
	if(x0 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	root = new (nothrow) float[data_size];
	if(root == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}


	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

//scaler parameter
float means[10] = { 10.0084, 9.9935  ,  9.8566, -400.2644 ,   4.9406 ,   9.9941 ,  10.0027,
    9.8779, -399.8569 ,   4.9322};
float scale[10] = { 6.0549 ,  6.0528 ,  6.1178, 115.715,    3.1917 ,  6.0553  , 6.0591 ,  6.1135,
 115.7213 ,  3.1881};
//

	// add data to the arrays
	float A_tmp, B_tmp, C_tmp, D_tmp, x0_tmp;
	int coeff_index = 0;
	while(coeff_index < data_size)
	{
		coeff_in_file >> A_tmp >> B_tmp >> C_tmp >> D_tmp >> x0_tmp;

		int indx = coeff_index%2;
		root 	[coeff_index]   = 0;
    A_coeff [coeff_index] = A_tmp;
    B_coeff [coeff_index] = B_tmp;
    C_coeff [coeff_index] = C_tmp;
    D_coeff [coeff_index] = D_tmp;
    x0 [coeff_index] = x0_tmp;
		A_coeff_half[coeff_index] 	= (A_tmp - means[indx*5 + 0])/scale[indx*5 +0];
		B_coeff_half	[coeff_index] 	= (B_tmp - means[indx*5 +1])/scale[indx*5 +1];
		C_coeff_half	[coeff_index]	= (C_tmp - means[indx*5 +2])/scale[indx*5 +2];
		D_coeff_half	[coeff_index] 	= (D_tmp - means[indx*5 +3])/scale[indx*5 +3];
		x0_half 		[coeff_index] = (x0_tmp - means[indx*5 +4])/scale[indx*5 +4];

    coeff_index++;
	}


	std::cout << "# Coefficients are read from file..." << std::endl;

	// memory allocations on the host
	float 	*A_coeff_d,
			*B_coeff_d,
			*C_coeff_d,
			*D_coeff_d,
			*x0_d;
  half 	*A_coeff_d_half,
    			*B_coeff_d_half,
    			*C_coeff_d_half,
    			*D_coeff_d_half,
    			*x0_d_half;

	float 	* root_d;

	cudaMalloc((void**) &A_coeff_d, data_size * sizeof(float));
	cudaMalloc((void**) &B_coeff_d, data_size * sizeof(float));
	cudaMalloc((void**) &C_coeff_d, data_size * sizeof(float));
	cudaMalloc((void**) &D_coeff_d, data_size * sizeof(float));
	cudaMalloc((void**) &x0_d, 		data_size * sizeof(float));

  cudaMalloc((void**) &A_coeff_d_half, data_size * sizeof(half));
	cudaMalloc((void**) &B_coeff_d_half, data_size * sizeof(half));
	cudaMalloc((void**) &C_coeff_d_half, data_size * sizeof(half));
	cudaMalloc((void**) &D_coeff_d_half, data_size * sizeof(half));
	cudaMalloc((void**) &x0_d_half, 		data_size * sizeof(half));

	cudaMalloc((void**) &root_d,	data_size * sizeof(float));

	std::cout << "# Memory allocation on GPU is done..." << std::endl;

	cudaMemcpy(A_coeff_d, A_coeff, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_coeff_d, B_coeff, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_coeff_d, C_coeff, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(D_coeff_d, D_coeff, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(x0_d, 	  x0,    data_size * sizeof(float), cudaMemcpyHostToDevice);



  cudaMemcpy(A_coeff_d_half, A_coeff_half, data_size * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(B_coeff_d_half, B_coeff_half, data_size * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(C_coeff_d_half, C_coeff_half, data_size * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(D_coeff_d_half, D_coeff_half, data_size * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(x0_d_half, 	  x0_half,    data_size * sizeof(half), cudaMemcpyHostToDevice);


		cudaMemcpyToSymbol(weight_1_half_d, &weight_1_half, 80 * sizeof(half));
		cudaMemcpyToSymbol(weight_2_half_d, &weight_2_half, 16 * sizeof(half));

		cudaMemcpyToSymbol(bias_1_half_d, &bias_1_half, 8 * sizeof(half));



	std::cout << "# Data are transfered to GPU..." << std::endl;

	dim3 dimBlock	( 256, 1 );
	dim3 dimGrid	( data_size / (256*2), 1 );
//	dim3 dimGrid	( 1, 1 );


	cudaEventRecord(start, 0);

#pragma parrot.start("nrpol3_kernel")

	nrpol3_kernel<<<dimGrid, dimBlock>>>(A_coeff_d, B_coeff_d, C_coeff_d, D_coeff_d, x0_d,
    A_coeff_d_half, B_coeff_d_half, C_coeff_d_half, D_coeff_d_half, x0_d_half
    , root_d, data_size, err_thresh);

#pragma parrot.end("nrpol3_kernel")

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
       	std::cout << "Something was wrong! Error code: " << cudaStatus << std::endl;
    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << "# Elapsed Time in `nrpoly3` kernel = " << elapsedTime << std::endl;
	std::cout << "# GPU computation is done ..." << std::endl;

	cudaMemcpy( root, root_d, data_size * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < data_size; i++)
	{
		root_out_file << root[i] << std::endl;
	}
/*
	for (int i =0; i<16; i++){
	//	if(i %16==0){
		for (int j=0; j<16; j++)
			printf("%f ", root[i*16+j]);
		printf("\n");
	//	}
	}
*/
	// close files
	root_out_file.close();
	coeff_in_file.close();

	// de-allocate the memory
	delete[] A_coeff;
	delete[] B_coeff;
	delete[] C_coeff;
	delete[] D_coeff;
	delete[] x0;
	delete[] root;

	// de-allocate cuda memory
	cudaFree(A_coeff_d);
	cudaFree(B_coeff_d);
	cudaFree(C_coeff_d);
	cudaFree(D_coeff_d);
	cudaFree(x0_d);
	cudaFree(root_d);

	std::cout << "Thank you..." << std::endl;
}
