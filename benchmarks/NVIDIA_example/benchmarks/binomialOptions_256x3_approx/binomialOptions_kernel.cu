/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// Global types and parameters
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_cuda.h>
#include "binomialOptions_common.h"
#include "realtype.h"

#include <mma.h>
#include <cuda_fp16.h>

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define SKEW_HALF 8 // from cuda sample avoid bank conflict

typedef struct
{
    real S;
    real X;
    real vDt;
    real puByDf;
    real pdByDf;
} __TOptionData;

TOptionData* d_OptionData;

real* d_CallValue;

__constant__  half weight_1_half_d[256];
__constant__ half weight_2_half_d[256];
__constant__ half weight_3_half_d[48];

__constant__ half bias_1_half_d[16];
__constant__ half bias_2_half_d[16];
__constant__ half bias_3_half_d[16];

__constant__ float means_d[15];
__constant__ float scale_d[15];

__device__ __inline__ half relu( half x){
  return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0) ;
}


//Preprocessed input option data


//static __constant__ __TOptionData d_OptionData[MAX_OPTIONS];
//static __device__           real d_CallValue[MAX_OPTIONS];



////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
#ifndef DOUBLE_PRECISION
__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
    float d = S * __expf(vDt * (2.0f * i - NUM_STEPS)) - X;
    return (d > 0.0F) ? d : 0.0F;
}
#else
__device__ inline double expiryCallValue(double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0.0) ? d : 0.0;
}
#endif


////////////////////////////////////////////////////////////////////////////////
// GPU kernel
////////////////////////////////////////////////////////////////////////////////
/*#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif
*/
using namespace nvcuda;
__global__ void binomialOptionsKernel(TOptionData* d_OptionData, real* d_CallValue)
{
/*
    const real      S = d_OptionData[blockIdx.x].S;
    const real      X = d_OptionData[blockIdx.x].X;
    const real    vDt = d_OptionData[blockIdx.x].vDt;
    const real puByDf = d_OptionData[blockIdx.x].puByDf;
    const real pdByDf = d_OptionData[blockIdx.x].pdByDf;
*/
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

  	//tensor approx
  	if(blockDim.x != 256) {
  		printf("not supported block dimension , it must be 256\n");
  		return;
  	}

  //	if (blockIdx.x >0) return ; //debug

  	__shared__ half A[8][256 + SKEW_HALF];
  	__shared__ half weight_1_shared[256];
  	__shared__ half bias_1_shared[256];

  	__shared__ half weight_2_shared[256];
  	__shared__ half bias_2_shared[256];

  	__shared__ half weight_3_shared[256];

  	__shared__ half bias_3_shared[256];

  	__shared__ half neuron_out[8][256 + SKEW_HALF]; //storing temp matrix mult res between layer

  	int row = threadIdx.x / 32;
  	int col = threadIdx.x % 16;

  	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
  //  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

  	int base_address = blockIdx.x*3*256; //number of elements processed for each block.
  	//remember , 256 threads to load values
  	weight_3_shared[threadIdx.x] = __float2half_rn(0.0);
  	for (int i = 0; i < 8 ; i ++ )
  		A[i][threadIdx.x] = __float2half_rn(0.0);
  	__syncthreads();
   	if (threadIdx.x <  16){
  		bias_1_shared[col] = bias_1_half_d[col];
  	} else if (threadIdx.x <  32){
  		bias_2_shared[col] = bias_2_half_d[col];
  	} else if (threadIdx.x < 48){
  		bias_3_shared[col] = bias_3_half_d[col];
  	} else if(threadIdx.x < 48 + 16){
  		for (int i = 0; i<3; i++)
  		weight_3_shared[col*16+i] = weight_3_half_d[col*3+i];
  	}

  	weight_1_shared[threadIdx.x] = weight_1_half_d[threadIdx.x];
  	weight_2_shared[threadIdx.x] = weight_2_half_d[threadIdx.x];



  	__syncthreads();

  //replicate
  	if (threadIdx.x >= 16){
  		bias_1_shared[threadIdx.x] =  bias_1_shared[col];
  		bias_2_shared[threadIdx.x] =  bias_2_shared[col];
  		bias_3_shared[threadIdx.x] =  bias_3_shared[col];
  	}


  for (int loop = 0; loop < 2; loop ++ ){ //each loop do the work of 8 warp

  if (threadIdx.x%32 < 16){

  	for (int k =0; k<3; k++)
  		A[row][16*col + k*5] = (d_OptionData[base_address + loop*128*3 + row*16*3  + col*3 + k].S - means_d[k*5])*scale_d[k*5];

  	for (int k =0; k<3; k++)
  		A[row][16*col + k*5+1] = (d_OptionData[base_address + loop*128*3 + row*16*3 + col*3 + k].X - means_d[k*5+1])*scale_d[k*5+1];

  	for (int k =0; k<3; k++)
  		A[row][16*col + k*5+2] = (d_OptionData[base_address + loop*128*3 + row*16*3 + col*3 + k].T - means_d[k*5+2])*scale_d[k*5+2];

  	for (int k =0; k<3; k++)
  		A[row][16*col + k*5+3] = 1.0;//d_OptionData[base_address + loop*128*3 + row*16*3 + col*3 + k].R;

  	for (int k =0; k<3; k++)
  		A[row][16*col + k*5+4] = -1.0;//d_OptionData[base_address + loop*128*3 + row*16*3 + col*3 + k].V;


  }
  	__syncthreads();
  //  if (row != 0) return;

  	wmma::load_matrix_sync(a_frag, (const __half*)A[row], 16);
  	wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared, 16);
  	wmma::load_matrix_sync(acc_frag, (const half*)&bias_1_shared, 16, wmma::mem_row_major);

  	wmma::mma_sync(c_frag, a_frag, b_frag, acc_frag);

  	for (int i = 0; i< c_frag.num_elements; i ++)
  		c_frag.x[i] = relu(c_frag.x[i]);

  //layer 2
  	wmma::store_matrix_sync((half*)neuron_out[row], c_frag, 16,wmma::mem_row_major);


  	wmma::load_matrix_sync(a_frag, (const __half*)neuron_out[row], 16);
  	wmma::load_matrix_sync(b_frag, (const __half*)&weight_2_shared, 16);
  	wmma::load_matrix_sync(acc_frag, (const half*)&bias_2_shared, 16, wmma::mem_row_major);

  	wmma::mma_sync(c_frag, a_frag, b_frag, acc_frag);
  	for (int i = 0; i< c_frag.num_elements; i ++)
  		c_frag.x[i] = relu(c_frag.x[i]);


  //layer 3
  	wmma::store_matrix_sync((half*)neuron_out[row], c_frag, 16,wmma::mem_row_major);

  	wmma::load_matrix_sync(a_frag, (const __half*)&neuron_out[row], 16);
  	wmma::load_matrix_sync(b_frag, (const __half*)&weight_3_shared, 16);
 // 	wmma::fill_fragment(acc_frag, 0.0f);
  	wmma::load_matrix_sync(acc_frag, (const __half*)&bias_3_shared, 16, wmma::mem_row_major);

  	wmma::mma_sync(c_frag, a_frag, b_frag, acc_frag);
//  	for (int i = 0; i< c_frag.num_elements; i ++)
//  		c_frag.x[i] = relu(c_frag.x[i]);//c_frag.x[i] = c_frag.x[i] + bias_3_half_d[0];
//	__syncthreads();
//    wmma::fill_fragment(c_frag, 1.0);
  //  if(row ==0)
  	wmma::store_matrix_sync((half*)&neuron_out[row], c_frag, 16,wmma::mem_row_major);

  //  if (blockIdx.x ==0)
  //  __syncthreads();
  //  d_CallValue[threadIdx.x] = neuron_out[0][threadIdx.x];

  //	angles[threadIdx.x] = A[1][threadIdx.x];

  //if not having this if condition, need to __syncthreads();

    if (threadIdx.x%32<16)
  	for (int k =0; k <3; k++){
  			d_CallValue[base_address + loop*128*3 + row*16*3 + col*3 + k] = relu(neuron_out[row][col*16+k]);
  		}

  	}




}

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU binomialOptions
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN,
    half* weight_1_half,
    half* bias_1_half,
    half* weight_2_half,
    half* bias_2_half,
    half* weight_3_half,
    half* bias_3_half
)
{
    //  TOptionData* h_OptionData;
    //  h_OptionData = (TOptionData*)malloc(MAX_OPTIONS*sizeof(TOptionData));
    //__TOptionData h_OptionData[MAX_OPTIONS];

    const float means[15] = {17.4525, 50.4189,  5.1461,  0.06,    0.1 ,   17.4719, 50.4323 , 5.1391,  0.06
  ,0.1 ,   17.4525, 50.4145,  5.1406,  0.06,    0.1};
    const float scale[15] = {7.2261e+00, 2.8553e+01, 2.8125e+00, 4.7483e-14, 1.0698e-13, 7.2277e+00,
 2.8545e+01, 2.8145e+00, 4.7483e-14 , 1.0698e-13,  7.2217e+00, 2.8542e+01,
 2.8131e+00, 4.7483e-14, 1.0698e-13};
    const float scale_mult[15] = {1.3839e-01,3.5023e-02, 3.5556e-01, 2.1060e+13, 9.3472e+12, 1.3836e-01,
 3.5033e-02, 3.5530e-01, 2.1060e+13, 9.3472e+12, 1.3847e-01, 3.5037e-02,
 3.5548e-01, 2.1060e+13, 9.3472e+12};
 /*
 for (int i = 0; i < optN; i++){
   int indx = i%3;

   h_OptionData[i].S      = (real)(optionData[i].S-means[indx*5 + 0])/scale[indx*5 + 0];
   h_OptionData[i].X      = (real)(optionData[i].X-means[indx*5 + 1])/scale[indx*5 + 1];
   h_OptionData[i].T    = (real)(optionData[i].T-means[indx*5 + 2])/scale[indx*5 + 2];
   h_OptionData[i].R = 1.0;//(real)(optionData[i].R-means[indx*5 + 3])/scale[indx*5 + 3];
   h_OptionData[i].V = -1.0;//(real)(optionData[i].V-means[indx*5 + 4])/scale[indx*5 + 4];
 }
 */
//  printf("\n%f %f %f %f %f\n",optionData[0].S,optionData[0].X,optionData[0].T,optionData[0].R,optionData[0].V );
/*
    for (int i = 0; i < optN; i++)
    {
        const real      T = optionData[i].T;
        const real      R = optionData[i].R;
        const real      V = optionData[i].V;

        const real     dt = T / (real)NUM_STEPS;
        const real    vDt = V * sqrt(dt);
        const real    rDt = R * dt;
        //Per-step interest and discount factors
        const real     If = exp(rDt);
        const real     Df = exp(-rDt);
        //Values and pseudoprobabilities of upward and downward moves
        const real      u = exp(vDt);
        const real      d = exp(-vDt);
        const real     pu = (If - d) / (u - d);
        const real     pd = (real)1.0 - pu;
        const real puByDf = pu * Df;
        const real pdByDf = pd * Df;

        int indx = i%3;

//        h_OptionData[i].S      = (real)optionData[i].S;
//        h_OptionData[i].X      = (real)optionData[i].X;
//        h_OptionData[i].vDt    = (real)vDt;
//        h_OptionData[i].puByDf = (real)puByDf;
//        h_OptionData[i].pdByDf = (real)pdByDf;

//scale
        h_OptionData[i].S      = (real)(optionData[i].S-means[indx*5 + 0])/scale[indx*5 + 0];
        h_OptionData[i].X      = (real)(optionData[i].X-means[indx*5 + 1])/scale[indx*5 + 1];
        h_OptionData[i].vDt    = (real)(vDt-means[indx*5 + 2])/scale[indx*5 + 2];
        h_OptionData[i].puByDf = (real)(puByDf-means[indx*5 + 3])/scale[indx*5 + 3];
        h_OptionData[i].pdByDf = (real)(pdByDf-means[indx*5 + 4])/scale[indx*5 + 4];

      //  if (i%100 ==0)
      //  printf("pass me %d %d\n\n",i,optN );

    }
*/
    cudaMemcpyToSymbol(weight_1_half_d, weight_1_half, 256 * sizeof(half));
		cudaMemcpyToSymbol(weight_2_half_d, weight_2_half, 256 * sizeof(half));
		cudaMemcpyToSymbol(weight_3_half_d, weight_3_half, 48 * sizeof(half));


		cudaMemcpyToSymbol(bias_1_half_d, bias_1_half, 16 * sizeof(half));
		cudaMemcpyToSymbol(bias_2_half_d, bias_2_half, 16 * sizeof(half));
		cudaMemcpyToSymbol(bias_3_half_d, bias_3_half, 16 * sizeof(half));
  checkCudaErrors( cudaMalloc((void**) &d_OptionData, optN * sizeof(TOptionData)));
  checkCudaErrors( cudaMalloc((void**) &d_CallValue, optN * sizeof(real)));

//    checkCudaErrors(cudaMemcpy(d_OptionData, h_OptionData, optN * sizeof(TOptionData), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionData, optionData, optN * sizeof(TOptionData), cudaMemcpyHostToDevice));

//    checkCudaErrors(cudaMemcpyToSymbol(d_OptionData, h_OptionData, optN * sizeof(__TOptionData)));
    checkCudaErrors(cudaMemcpyToSymbol(means_d, means, 15 * sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(scale_d, scale_mult, 15 * sizeof(float)));

//    binomialOptionsKernel<<<optN, THREADBLOCK_SIZE>>>();
    binomialOptionsKernel<<<optN/(256*3) +1, 256>>>(d_OptionData, d_CallValue);
//      binomialOptionsKernel<<<1, 256>>>(d_OptionData, d_CallValue);
//    getLastCudaError("binomialOptionsKernel() execution failed.\n");
    //checkCudaErrors(cudaMemcpyFromSymbol(callValue, d_CallValue, optN *sizeof(real)));
    cudaMemcpy( callValue, d_CallValue, optN * sizeof(real), cudaMemcpyDeviceToHost);
     getLastCudaError("binomialOptionsKernel() execution failed.\n");



}
