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

//__half2_raw one; one.x = 0x3C00; one.y = 0x3C00;

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(- R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


__device__ __inline__ half relu( half x){
  return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0) ;
}

__global__ void BlackScholesGPU(
    half *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    half *d_StockPrice_half,
    half *d_OptionStrike_half,
    half *d_OptionYears_half,
    float Riskfree,
    float Volatility,
    int optN,
    int speed
)
{
  if (blockIdx.x %10 < speed){

   // const int opt = blockDim.x * blockIdx.x + threadIdx.x;
  // if(blockIdx.x != 0 ) return;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	half2 bias_2_dev = __floats2half2_rn(0.0829115, 0.114646);
   int real_tid =  threadIdx.x;
    int warp_id = real_tid /32;
    int warp_lane = real_tid %32;

    __shared__ half weight_1_shared[128];
    __shared__ half bias_1_shared[256];
    __shared__ half neuron_out[8][512];

    if (real_tid <48){
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
			  neuron_out[warp_id][warp_lane+ 0*32] = (d_StockPrice_half[2*tid] - mean_d[0])*scale_d[0];
			  neuron_out[warp_id][warp_lane+ 1*32] = (d_OptionStrike_half[2*tid] - mean_d[1])*scale_d[1];
			  neuron_out[warp_id][warp_lane+ 2*32] = (d_OptionYears_half[2*tid] - mean_d[2])*scale_d[2];

			  neuron_out[warp_id][warp_lane+ 3*32] = (d_StockPrice_half[2*tid+1] - mean_d[3])*scale_d[3];
			  neuron_out[warp_id][warp_lane+ 4*32] = (d_OptionStrike_half[2*tid+1] - mean_d[4])*scale_d[4];
			  neuron_out[warp_id][warp_lane+ 5*32] = (d_OptionYears_half[2*tid+1] - mean_d[5])*scale_d[5];

		//	  neuron_out[warp_id][warp_lane+ 3*32] = x[tid+2];
		// }

  		wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
  		wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared, 8);
  		wmma::load_matrix_sync(c_frag, (const half*)&bias_1_shared, 32, wmma::mem_col_major);
  //    wmma::fill_fragment(c_frag, 0.0);
  		wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
  		for (int i = 0; i< c_frag.num_elements; i ++)
  			c_frag.x[i] = relu(c_frag.x[i]);
 		wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);
  //  wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 8,wmma::mem_row_major);



		__syncwarp();
		half2 output1= __float2half2_rn(0.0);
		for (int i =0 ; i <8 ; i++ )
			output1 += __half2half2(neuron_out[warp_id][warp_lane+i*32])*weight_2_half_d[i];
		output1 += bias_2_dev;//__float2half_rn(BIAS2);

  if (2*tid < optN) {
	  d_CallResult[2*tid] = output1.x;//__half2float(output1.x);
	  d_CallResult[2*tid+1] = output1.y;//__half2float(output1.y);

  }

} else { //run float version
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  float callResult1, callResult2;
  float putResult1, putResult2;
  BlackScholesBodyGPU(
      callResult1,
      putResult1,
      d_StockPrice[2*tid],
      d_OptionStrike[2*tid],
      d_OptionYears[2*tid],
      Riskfree,
      Volatility
  );
  BlackScholesBodyGPU(
      callResult2,
      putResult2,
      d_StockPrice[2*tid+1],
      d_OptionStrike[2*tid+1],
      d_OptionYears[2*tid+1],
      Riskfree,
      Volatility
  );
  d_CallResult[2*tid] = callResult1;//__half2float(output1.x);
  d_CallResult[2*tid+1] = callResult2;//__half2float(output1.y);


  }




}
