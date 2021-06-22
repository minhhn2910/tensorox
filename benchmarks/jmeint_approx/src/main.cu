// Designed by: Amir Yazdanbakhsh
// Date: March 26th - 2015
// Alternative Computing Technologies Lab.
// Georgia Institute of Technology


#include "stdlib.h"
#include <fstream>
#include <iostream>
#include <cstddef>

// Cuda Libraries
#include <cuda_runtime_api.h>
#include <cuda.h>

#define EPSILON 1e-12 // EPSILON represents the error buffer used to denote a hit

using namespace std;


#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)


// The only dimensions currently supported by WMMA

//__constant__ half weight_1_half_d[2][256]; //[32x16]
__constant__ half weight_1_half_d[144];
__constant__ half weight_2_half_d[8];


__constant__ half bias_1_half_d[8];
//__constant__ half bias_3_half_d[1]; only 1 value, no need
#define BIAS_2 0.0617521

__device__ __inline__ half relu( half x){
  return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0) ;
}

__device__ __inline__ half relu_last( half x){
  return (x>=__float2half_rn(0.5))? __float2half_rn(1.0) :__float2half_rn(0.0) ;
}

#define CONST_SCALE __float2half_rn(0.033)

__global__ void jmeint_kernel(float *v0_d, float *v1_d, float *v2_d, float *u0_d, float*u1_d, float*u2_d,
  half *v0_d_half, half *v1_d_half, half *v2_d_half, half *u0_d_half, half *u1_d_half, half *u2_d_half
  , bool* intersect_d, int size)
{

  //  if (blockIdx.x != 0)
  //    return;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid;
    int real_tid =  threadIdx.x;
     int warp_id = real_tid /32;
     int warp_lane = real_tid %32;

     __shared__ half A[2][8][512];
     __shared__ half weight_1_shared[2][128];
     __shared__ half bias_1_shared[256];
  //   __shared__ half neuron_out[8][512];

     weight_1_shared[0][threadIdx.x] = weight_1_half_d[threadIdx.x];//__float2half_rn(0.0);
   	if(threadIdx.x < 16)
   			weight_1_shared[1][threadIdx.x] = weight_1_half_d[128+threadIdx.x];


     //simple trick no need if/else, read as col_major later
 	   bias_1_shared[real_tid] = bias_1_half_d[warp_id];

     for (int i = 0; i<8 ; i++){
       A[1][i][real_tid] = 0.0;
       A[1][i][real_tid+256] = 0.0;
     }
     __syncthreads();

 	//	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
 		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag_col;
 		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
 	//	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
 		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

 	//	if (tid < nthreads-1){
 			  A[0][warp_id][warp_lane+ 0*32] = v0_d_half[idx * 3 + 0]*CONST_SCALE;
 			  A[0][warp_id][warp_lane+ 1*32] = v0_d_half[idx * 3 + 1]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 2*32] = v0_d_half[idx * 3 + 2]*CONST_SCALE;

        A[0][warp_id][warp_lane+ 3*32] = v1_d_half[idx * 3 + 0]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 4*32] = v1_d_half[idx * 3 + 1]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 5*32] = v1_d_half[idx * 3 + 2]*CONST_SCALE;

        A[0][warp_id][warp_lane+ 6*32] = v2_d_half[idx * 3 + 0]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 7*32] = v2_d_half[idx * 3 + 1]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 8*32] = v2_d_half[idx * 3 + 2]*CONST_SCALE;

        A[0][warp_id][warp_lane+ 9*32] = u0_d_half[idx * 3 + 0]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 10*32] = u0_d_half[idx * 3 + 1]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 11*32] = u0_d_half[idx * 3 + 2]*CONST_SCALE;

        A[0][warp_id][warp_lane+ 12*32] = u1_d_half[idx * 3 + 0]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 13*32] = u1_d_half[idx * 3 + 1]*CONST_SCALE;
        A[0][warp_id][warp_lane+ 14*32] = u1_d_half[idx * 3 + 2]*CONST_SCALE;

        A[0][warp_id][warp_lane+ 15*32] = u2_d_half[idx * 3 + 0]*CONST_SCALE;
        //next tensor
        A[1][warp_id][warp_lane+ 0*32] = u2_d_half[idx * 3 + 1]*CONST_SCALE;
        A[1][warp_id][warp_lane+ 1*32] = u2_d_half[idx * 3 + 2]*CONST_SCALE;
        __syncthreads();
 		// }
   		wmma::load_matrix_sync(a_frag_col, (const __half*)A[0][warp_id], 32);
   		wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared[0], 8);
   		wmma::load_matrix_sync(c_frag, (const half*)&bias_1_shared, 32, wmma::mem_col_major);

   		wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);

  		//wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);
      wmma::store_matrix_sync((half*)A[0][warp_id], c_frag, 32,wmma::mem_col_major);


      wmma::load_matrix_sync(a_frag_col, (const __half*)A[1][warp_id], 32);
      wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared[1], 8);

      wmma::load_matrix_sync(c_frag, (const half*)&A[0][warp_id], 32, wmma::mem_col_major);
      wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);

      for (int i = 0; i< c_frag.num_elements; i ++)
   			c_frag.x[i] = relu(c_frag.x[i]);

//debug
     wmma::store_matrix_sync((half*)A[0][warp_id], c_frag, 32,wmma::mem_col_major);

      __syncwarp();
      half output1= __float2half_rn(0.0);
      for (int i =0 ; i <8 ; i++ )
        output1 += A[0][warp_id][warp_lane+i*32]*weight_2_half_d[i];
      output1 += __float2half_rn(BIAS_2);
      __syncwarp();

    if (output1 >= __float2half_rn(0.5))
      intersect_d[tid] = true;
    else
      intersect_d[tid] = false;
    //  debug[threadIdx.x] = output1;//neuron_out[0][threadIdx.x];


/*
 		__syncwarp();
 		half output1= __float2half_rn(0.0);
 		for (int i =0 ; i <8 ; i++ )
 			output1 += neuron_out[warp_id][warp_lane+i*32]*weight_2_half_d[i];
 		output1 += __float2half_rn(BIAS2);

   if (tid < nthreads-1) {
     y[tid] = __half2float(output1)*1e-6;

   }
*/
}

float weight_1[144] = {0.511078, 0.523341, 0.066438, -0.491299, 0.443703, 0.942849, 0.596723, -0.319132, 0.449167, 0.255444, 0.473568, -0.39083, -0.371555, 0.0453213, 0.0861016, 0.280701, -0.421177, -0.14251, 0.0308778, -0.429457, 0.400507, 0.14804, -0.00634764, 0.356105, 0.416433, -0.136496, -0.0936358, -0.356244, 0.388306, -0.245637, -0.211574, -0.352305, 0.398325, -0.818396, 0.149469, -0.497483, -0.405287, 0.423554, -0.269324, 0.476809, -0.411678, -0.121283, -0.000869494, -0.47774, 0.4155, 0.33816, 0.0341384, 0.387186, 0.461407, 0.335607, -0.0176836, -0.307993, 0.407101, -0.202389, -0.0277943, -0.407035, 0.471145, 0.00753348, 0.449523, -0.432606, -0.372076, 0.273333, -0.0572681, 0.367748, -0.407902, -0.148693, 0.00476071, -0.469246, 0.410727, 0.282667, 0.0214304, 0.385739, -0.481306, -0.251702, -0.082988, 0.387699, -0.434762, -0.138973, -0.103004, 0.380288, -0.453013, 0.219075, 0.360367, 0.445675, 0.389814, -0.227433, -0.0223918, -0.395539, 0.419846, 0.187924, 0.0673863, 0.460819, -0.401764, -0.251448, -0.0710916, -0.353779, -0.479561, -0.214291, -0.0929197, 0.372163, -0.421492, -0.063086, -0.0723261, 0.397593, -0.446763, 0.297718, 0.415162, 0.447785, 0.382669, -0.264714, 0.0129754, -0.404251, 0.413612, 0.14786, 0.0472674, 0.471104, -0.395543, -0.287225, -0.0680886, -0.359023, -0.473956, -0.220853, -0.0993806, 0.410354, -0.403897, -0.14258, -0.0901898, 0.379739, -0.450525, 0.260478, 0.399954, 0.44896, 0.400013, -0.243192, 0.0139935, -0.401857, 0.42331, 0.191439, 0.0510963, 0.471557, -0.409442, -0.21915, -0.0929248, -0.349474};
float bias_1[8] = {0.00521765, -0.0327343, -0.125869, -0.00520308, -0.019705, -0.0615173, 1.24405, -0.0105536};
float weight_2[8] = {0.207281, -0.0984865, 0.0587332, 0.220695, 0.225964, -0.103368, 0.101314, 0.259321};
half weight_1_half[144],bias_1_half[8], weight_2_half[8];

void prepare_weights(){
  for (int i=0;i < 144; i++)
    weight_1_half[i] = __float2half_rn(weight_1[i]);
  for (int i=0;i < 8; i++){
    bias_1_half[i] = __float2half_rn(bias_1[i]);
    weight_2_half[i] = __float2half_rn(weight_2[i]);
  }

}

int main(int argc, char* argv[])
{
	if(argc != 3)
	{
		std::cerr << "Usage: ./jmeint.out <input file locations> <output file>" << std::endl;
		exit(EXIT_FAILURE);
	}

  prepare_weights();

	float (*v0)[3];
	float (*v1)[3];
	float (*v2)[3];
	float (*u0)[3];
	float (*u1)[3];
	float (*u2)[3];

  half (*v0_half)[3];
	half (*v1_half)[3];
	half (*v2_half)[3];
	half (*u0_half)[3];
	half (*u1_half)[3];
	half (*u2_half)[3];

	bool  *intersect;
//  float  *intersect; //debug

	cudaError_t cudaStatus;

	int data_size = 0;

	// process the files
	ifstream locations_in_file (argv[1]);
	ofstream intersect_out_file (argv[2]);


	if(locations_in_file.is_open())
	{
		locations_in_file >> data_size;
		std::cout << "# Data Size = " << data_size << std::endl;
	}

//  intersect = new (nothrow) float[data_size]; //debug
	intersect = new (nothrow) bool[data_size];

	// allocate the memory
	v0 = new (nothrow) float[data_size][3];

  v0_half = new (nothrow) half[data_size][3];
	if(v0 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	v1 = new (nothrow) float[data_size][3];
  v1_half = new (nothrow) half[data_size][3];
	if(v1 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	v2 = new (nothrow) float[data_size][3];
  v2_half = new (nothrow) half[data_size][3];
	if(v2 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	u0 = new (nothrow) float[data_size][3];
  u0_half = new (nothrow) half[data_size][3];
	if(u0 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	u1 = new (nothrow) float[data_size][3];
  u1_half = new (nothrow) half[data_size][3];
	if(u1 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	u2 = new (nothrow) float[data_size][3];
  u2_half = new (nothrow) half[data_size][3];
	if(u2 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}



	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// add data to the arrays
	int loc_index = 0;
	while(loc_index < data_size)
	{
		locations_in_file 	>> v0[loc_index][0] >> v0[loc_index][1] >> v0[loc_index][2]
							>> v1[loc_index][0] >> v1[loc_index][1] >> v1[loc_index][2]
							>> v2[loc_index][0] >> v2[loc_index][1] >> v2[loc_index][2]
							>> u0[loc_index][0] >> u0[loc_index][1] >> u0[loc_index][2]
							>> u1[loc_index][0] >> u1[loc_index][1] >> u1[loc_index][2]
							>> u2[loc_index][0] >> u2[loc_index][1] >> u2[loc_index][2];

    v0_half[loc_index][0] = __float2half_rn(v0[loc_index][0]);
    v0_half[loc_index][1] = __float2half_rn(v0[loc_index][1]);
    v0_half[loc_index][2] = __float2half_rn(v0[loc_index][2]);

    v1_half[loc_index][0] = __float2half_rn(v1[loc_index][0]);
    v1_half[loc_index][1] = __float2half_rn(v1[loc_index][1]);
    v1_half[loc_index][2] = __float2half_rn(v1[loc_index][2]);

    v2_half[loc_index][0] = __float2half_rn(v2[loc_index][0]);
    v2_half[loc_index][1] = __float2half_rn(v2[loc_index][1]);
    v2_half[loc_index][2] = __float2half_rn(v2[loc_index][2]);


    u0_half[loc_index][0] = __float2half_rn(u0[loc_index][0]);
    u0_half[loc_index][1] = __float2half_rn(u0[loc_index][1]);
    u0_half[loc_index][2] = __float2half_rn(u0[loc_index][2]);

    u1_half[loc_index][0] = __float2half_rn(u1[loc_index][0]);
    u1_half[loc_index][1] = __float2half_rn(u1[loc_index][1]);
    u1_half[loc_index][2] = __float2half_rn(u1[loc_index][2]);

    u2_half[loc_index][0] = __float2half_rn(u2[loc_index][0]);
    u2_half[loc_index][1] = __float2half_rn(u2[loc_index][1]);
    u2_half[loc_index][2] = __float2half_rn(u2[loc_index][2]);


		loc_index++;
	}


	std::cout << "# Coordinates are read from file..." << std::endl;

	// memory allocations on the host
	float *v0_d;
	float *v1_d;
	float *v2_d;
	float *u0_d;
	float *u1_d;
	float *u2_d;

  half *v0_d_half;
  half *v1_d_half;
  half *v2_d_half;
  half *u0_d_half;
  half *u1_d_half;
  half *u2_d_half;

	bool  *intersect_d;
//  float  *intersect_d; //debug


	cudaMalloc((void**) &v0_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &v1_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &v2_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &u0_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &u1_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &u2_d, data_size * 3 * sizeof(float));

  cudaMalloc((void**) &v0_d_half, data_size * 3 * sizeof(half));
	cudaMalloc((void**) &v1_d_half, data_size * 3 * sizeof(half));
	cudaMalloc((void**) &v2_d_half, data_size * 3 * sizeof(half));
	cudaMalloc((void**) &u0_d_half, data_size * 3 * sizeof(half));
	cudaMalloc((void**) &u1_d_half, data_size * 3 * sizeof(half));
	cudaMalloc((void**) &u2_d_half, data_size * 3 * sizeof(half));


//debug
  float* debug_h, *debug_d;
  debug_h = (float*)malloc(data_size * sizeof(float));
  cudaMalloc((void**) &debug_d, data_size * sizeof(float));


	cudaMalloc((void**) &intersect_d, data_size * sizeof(bool));
//  cudaMalloc((void**) &intersect_d, data_size * sizeof(float));//debug


	std::cout << "# Memory allocation on GPU is done..." << std::endl;

	cudaMemcpy(v0_d, v0, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(v1_d, v1, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(v2_d, v2, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(u0_d, u0, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(u1_d, u1, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(u2_d, u2, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(v0_d_half, v0_half, data_size * 3 * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(v1_d_half, v1_half, data_size * 3 * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(v2_d_half, v2_half, data_size * 3 * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(u0_d_half, u0_half, data_size * 3 * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(u1_d_half, u1_half, data_size * 3 * sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(u2_d_half, u2_half, data_size * 3 * sizeof(half), cudaMemcpyHostToDevice);



  cudaMemcpyToSymbol(weight_1_half_d, &weight_1_half, 144 * sizeof(half));
  cudaMemcpyToSymbol(weight_2_half_d, &weight_2_half, 8 * sizeof(half));
  cudaMemcpyToSymbol(bias_1_half_d, &bias_1_half, 8 * sizeof(half));



	std::cout << "# Data are transfered to GPU..." << std::endl;

	dim3 dimBlock	( 256, 1 );
	dim3 dimGrid	( data_size / 256, 1 );
//	dim3 dimGrid	(1, 1 ); //debug


	cudaEventRecord(start, 0);

#pragma parrot.start("jmeint_kernel")

	jmeint_kernel<<<dimGrid, dimBlock>>>(v0_d, v1_d, v2_d, u0_d, u1_d, u2_d,
    v0_d_half, v1_d_half, v2_d_half, u0_d_half, u1_d_half, u2_d_half
     , intersect_d, data_size);

#pragma parrot.end("jmeint_kernel")

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

	std::cout << "# Elapsed Time in `jmeint` kernel = " << elapsedTime << std::endl;
	std::cout << "# GPU computation is done ..." << std::endl;

	cudaMemcpy(intersect, intersect_d, data_size * sizeof(bool), cudaMemcpyDeviceToHost);
//  cudaMemcpy(intersect, intersect_d, data_size * sizeof(float), cudaMemcpyDeviceToHost); //debug




	for(int i = 0; i < data_size; i++)
	{
		intersect_out_file << intersect[i];
		intersect_out_file << std::endl;
	}

	// close files
	locations_in_file.close();
	intersect_out_file.close();

	// de-allocate the memory
	delete[] v0;
	delete[] v1;
	delete[] v2;
	delete[] u0;
	delete[] u1;
	delete[] u2;
	delete[] intersect;

	// de-allocate cuda memory
	cudaFree(v0_d);
	cudaFree(v1_d);
	cudaFree(v2_d);
	cudaFree(u0_d);
	cudaFree(u1_d);
	cudaFree(u2_d);
	cudaFree(intersect_d);

	std::cout << "Thank you..." << std::endl;
}
