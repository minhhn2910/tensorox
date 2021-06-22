/**
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
// Global types
////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include <curand_kernel.h>
#include "MonteCarlo_common.h"

////////////////////////////////////////////////////////////////////////////////
// Helper reduction template
// Please see the "reduction" CUDA Sample for more information
////////////////////////////////////////////////////////////////////////////////
#include "MonteCarlo_reduction.cuh"

////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side data structures
////////////////////////////////////////////////////////////////////////////////
#define MAX_OPTIONS (1024*1024)

//Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real MuByT;
    real VBySqrtT;
} __TOptionData;


#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

__constant__ half weight_1_half_d[128];
__constant__ half weight_2_half_d[64];
__constant__ half bias_1_half_d[8];
__constant__ half bias_2_half_d[8];

__constant__ float mean_d[16];
__constant__ float scale_d[16];

__device__ __inline__ half relu( half x){
  return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0) ;
}

#define THREAD_N 256

////////////////////////////////////////////////////////////////////////////////
// This kernel computes the integral over all paths using a single thread block
// per option. It is fastest when the number of thread blocks times the work per
// block is high enough to keep the GPU busy.
////////////////////////////////////////////////////////////////////////////////
static __global__ void MonteCarloOneBlockPerOption(
    curandState * __restrict rngStates,
    const __TOptionData * __restrict d_OptionData,
    __TOptionValue * __restrict d_CallValue,
    int pathN,
    int optionN)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
   int idx = tid;
  //tensor approx
  if(blockDim.x != 256) {
    printf("not supported block dimension , it must be 256\n");
    return;
  }

  int real_tid =  threadIdx.x;
int warp_id = real_tid /32;
int warp_lane = real_tid %32;
__shared__ half neuron_out[8][512];
__shared__ half weight_1_shared[128];
__shared__ half bias_1_shared[256];

__shared__ half weight_2_shared[128];
__shared__ half bias_2_shared[256];

   weight_1_shared[real_tid] = weight_1_half_d[real_tid];
   weight_2_shared[real_tid] = 0.0;
   for (int i = 0; i<8 ; i++){
     neuron_out[i][real_tid] = 0.0;
     neuron_out[i][real_tid+256] = 0.0;
   }
   __syncthreads();
  if (real_tid < 64){
    weight_2_shared[real_tid] = weight_2_half_d[real_tid];;
  }

  //simple trick no need if/else, read as col_major later
  bias_1_shared[real_tid] = bias_1_half_d[warp_id];
  bias_2_shared[real_tid] = bias_2_half_d[warp_id];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag_col;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    neuron_out[warp_id][warp_lane+ 0*32] = (d_OptionData[4*tid].S - mean_d[0])*scale_d[0];
    neuron_out[warp_id][warp_lane+ 1*32] = (d_OptionData[4*tid].X - mean_d[1])*scale_d[1];
   	neuron_out[warp_id][warp_lane+ 2*32] = (d_OptionData[4*tid].MuByT - mean_d[2])*scale_d[2];
    neuron_out[warp_id][warp_lane+ 3*32] = (d_OptionData[4*tid].VBySqrtT - mean_d[3])*scale_d[3];

    neuron_out[warp_id][warp_lane+ 4*32] = (d_OptionData[4*tid+1].S - mean_d[4])*scale_d[4];
    neuron_out[warp_id][warp_lane+ 5*32] = (d_OptionData[4*tid+1].X - mean_d[5])*scale_d[5];
   	neuron_out[warp_id][warp_lane+ 6*32] = (d_OptionData[4*tid+1].MuByT - mean_d[6])*scale_d[6];
    neuron_out[warp_id][warp_lane+ 7*32] = (d_OptionData[4*tid+1].VBySqrtT - mean_d[7])*scale_d[7];

    neuron_out[warp_id][warp_lane+ 8*32] = (d_OptionData[4*tid+2].S - mean_d[8])*scale_d[8];
    neuron_out[warp_id][warp_lane+ 9*32] = (d_OptionData[4*tid+2].X - mean_d[9])*scale_d[9];
   	neuron_out[warp_id][warp_lane+ 10*32] = (d_OptionData[4*tid+2].MuByT - mean_d[10])*scale_d[10];
    neuron_out[warp_id][warp_lane+ 11*32] = (d_OptionData[4*tid+2].VBySqrtT - mean_d[11])*scale_d[11];

    neuron_out[warp_id][warp_lane+ 12*32] = (d_OptionData[4*tid+3].S - mean_d[12])*scale_d[12];
    neuron_out[warp_id][warp_lane+ 13*32] = (d_OptionData[4*tid+3].X - mean_d[13])*scale_d[13];
   	neuron_out[warp_id][warp_lane+ 14*32] = (d_OptionData[4*tid+3].MuByT - mean_d[14])*scale_d[14];
    neuron_out[warp_id][warp_lane+ 15*32] = (d_OptionData[4*tid+3].VBySqrtT - mean_d[15])*scale_d[15];

    __syncthreads();

    wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
     wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared, 8);
     wmma::load_matrix_sync(c_frag, (const half*)&bias_1_shared, 32, wmma::mem_col_major);

     wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
     for (int i = 0; i< c_frag.num_elements; i ++)
       c_frag.x[i] = relu(c_frag.x[i]);
     wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);

     wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
     wmma::load_matrix_sync(b_frag, (const __half*)&weight_2_shared, 8);
     wmma::load_matrix_sync(c_frag, (const half*)&bias_2_shared, 32, wmma::mem_col_major);

     wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
     for (int i = 0; i< c_frag.num_elements; i ++)
       c_frag.x[i] = relu(c_frag.x[i]);
     wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);
     __syncwarp();
     d_CallValue[4*tid+0].Expected = neuron_out[warp_id][warp_lane+0*32];
     d_CallValue[4*tid+1].Expected = neuron_out[warp_id][warp_lane+1*32];
     d_CallValue[4*tid+2].Expected = neuron_out[warp_id][warp_lane+2*32];
     d_CallValue[4*tid+3].Expected = neuron_out[warp_id][warp_lane+3*32];
}

static __global__ void rngSetupStates(
    curandState *rngState,
    int device_id)
{
    // determine global thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Each threadblock gets different seed,
    // Threads within a threadblock get different sequence numbers
    curand_init(blockIdx.x + gridDim.x * device_id, threadIdx.x, 0, &rngState[tid]);
}



////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU Monte Carlo
////////////////////////////////////////////////////////////////////////////////
extern "C" void initMonteCarloGPU(TOptionPlan *plan )
{
  float weight_1[128] = {1.05637, -0.00168668, 2.99156, 0.00197077, 0.000549778, 0.984048, 0.0800636, -0.000238908, -0.176982, 0.000201989, -0.903573, -0.00143033, 0.00176847, -0.164562, -0.0171022, 5.62609e-05, 0.0513151, -0.00514326, 0.158978, 0.00869174, -0.00897845, 0.068531, 0.017544, -0.00168464, 0.0842414, 0.00833916, 0.0399934, -0.00868932, 0.00771363, 0.06366, -0.00353415, 0.000128761, 0.473163, -0.00161178, 0.00233536, 0.00251595, -0.000346638, 0.0469293, 0.81149, 3.25743, -0.0811349, 0.000900033, -0.000710369, 0.00157221, 0.00270015, -0.00891659, -0.138095, -0.983193, 0.0118137, -0.000574662, 0.00774465, -0.00544387, 0.00892755, -0.0101543, 0.0278426, 0.180797, 0.0484582, -0.00300508, -0.0071916, 0.00251435, -0.0001952, 0.0146376, 0.0739883, 0.0314767, -0.114392, 2.88635, 0.000591933, 6.6453e-05, 0.00107987, 0.760334, 0.704939, 0.000270815, 0.0185912, -0.87244, 0.00177448, 0.00119561, -0.00102587, -0.126177, -0.117979, -7.60254e-05, -0.00480397, 0.143811, -0.00277964, -0.00416187, 0.00587754, 0.0431332, 0.0383064, 0.00197552, -0.0122519, 0.0466076, 0.00338718, 0.0114404, -0.00493077, 0.0492921, 0.0552189, -0.00132913, -0.00403811, 0.000989323, 0.000111006, 1.5035, 3.21473, 0.0202481, 0.0501683, -0.00151138, -0.000490921, -0.000433085, -0.000418472, -0.277786, -0.97519, -0.00383735, -0.00745126, 0.00257238, 0.00721398, 0.00600034, -0.000219815, 0.000112429, 0.213265, -0.00676159, 0.00315878, -0.00727925, -0.0074255, -0.00959887, 0.00224475, 0.200172, -0.0219922, 0.00954229, -0.000248317, 0.0114735};
  float bias_1 [8] = {2.83244, 2.67474, 2.77911, 2.489, 2.93423, 3.15722, 2.63993, 3.02443};
  float weight_2[32] = {1.35478, 1.46346, -1.84121, 0.00283766, 0.000341243, -0.00175773, 3.67062, 0.00163694, 3.54323, -0.000925917, -0.00100888, -0.000600053, 0.0210331, -0.0429193, -0.064157, 1.89791, 0.000652647, 0.00345044, 0.00265811, 3.13617, 0.999721, -1.75755, 1.90735, -0.000244988, -0.855003, 2.13892, 0.969283, -0.00173488, 0.000638993, 3.24544, 0.00160997, 0.0010467};
  float bias_2 [8] = {-1.71243, -1.0659, -0.141981, -1.00217};

  const float means[16] = {27.5189, 17.501,   0.165,   0.1697, 27.4871, 17.5025,  0.1651,  0.1697, 27.4734,
17.4909,  0.1648,  0.1696, 27.5205, 17.4989,  0.1651,  0.1697};
  const float scale_mult[16] = {0.0768,  0.2308, 15.7547, 28.7474,  0.077,   0.2313, 15.7361, 28.7148,  0.077,
0.2311, 15.7267, 28.692,   0.077,   0.231,  15.7363 ,28.7232};



  half weight_1_half[128], bias_1_half[8], weight_2_half[64], bias_2_half[8];

  for (int i =0 ; i< 8 ; i ++){
  		bias_1_half[i] = __float2half_rn(bias_1[i]);
  		bias_2_half[i] = __float2half_rn(bias_2[i]);
  	}

  	for (int i = 0 ; i<128; i++){
  		weight_1_half[i] = __float2half_rn(weight_1[i]);

  	}
    for (int i =0; i <8; i ++){
          for (int j =0 ; j <4 ; j ++)
            weight_2_half[i*8+j] = __float2half_rn(weight_2[i*4 + j]);
          for (int j = 4; j <8 ;j++)
            weight_2_half[i*8+j] = __float2half_rn(0.0);
    }

      cudaMemcpyToSymbol(weight_1_half_d, weight_1_half, 128 * sizeof(half));
  		cudaMemcpyToSymbol(weight_2_half_d, weight_2_half, 64 * sizeof(half));


  		cudaMemcpyToSymbol(bias_1_half_d, bias_1_half, 8 * sizeof(half));
  		cudaMemcpyToSymbol(bias_2_half_d, bias_2_half, 8 * sizeof(half));

      checkCudaErrors(cudaMemcpyToSymbol(mean_d, means, 16 * sizeof(float)));
      checkCudaErrors(cudaMemcpyToSymbol(scale_d, scale_mult, 16 * sizeof(float)));


      checkCudaErrors(cudaMalloc(&plan->d_OptionData, sizeof(__TOptionData)*(plan->optionCount)));
      checkCudaErrors(cudaMalloc(&plan->d_CallValue, sizeof(__TOptionValue)*(plan->optionCount)));
      checkCudaErrors(cudaMallocHost(&plan->h_OptionData, sizeof(__TOptionData)*(plan->optionCount)));
      //Allocate internal device memory
      checkCudaErrors(cudaMallocHost(&plan->h_CallValue, sizeof(__TOptionValue)*(plan->optionCount)));
      //Allocate states for pseudo random number generators
/*
      checkCudaErrors(cudaMalloc((void **) &plan->rngStates,
                                 plan->gridSize * THREAD_N * sizeof(curandState)));

      checkCudaErrors(cudaMemset(plan->rngStates, 0.0, plan->gridSize * THREAD_N * sizeof(curandState)));

      // place each device pathN random numbers apart on the random number sequence
      rngSetupStates<<<plan->gridSize, THREAD_N>>>(plan->rngStates, plan->device);
      getLastCudaError("rngSetupStates kernel failed.\n");
*/
}
/*extern "C" void initMonteCarloGPU(TOptionPlan *plan, )
{
    checkCudaErrors(cudaMalloc(&plan->d_OptionData, sizeof(__TOptionData)*(plan->optionCount)));
    checkCudaErrors(cudaMalloc(&plan->d_CallValue, sizeof(__TOptionValue)*(plan->optionCount)));
    checkCudaErrors(cudaMallocHost(&plan->h_OptionData, sizeof(__TOptionData)*(plan->optionCount)));
    //Allocate internal device memory
    checkCudaErrors(cudaMallocHost(&plan->h_CallValue, sizeof(__TOptionValue)*(plan->optionCount)));
    //Allocate states for pseudo random number generators
    checkCudaErrors(cudaMalloc((void **) &plan->rngStates,
                               plan->gridSize * THREAD_N * sizeof(curandState)));
    checkCudaErrors(cudaMemset(plan->rngStates, 0.0, plan->gridSize * THREAD_N * sizeof(curandState)));

    // place each device pathN random numbers apart on the random number sequence
    rngSetupStates<<<plan->gridSize, THREAD_N>>>(plan->rngStates, plan->device);
    getLastCudaError("rngSetupStates kernel failed.\n");
}
*/

//Compute statistics and deallocate internal device memory
extern "C" void closeMonteCarloGPU(TOptionPlan *plan)
{
/*
    for (int i = 0; i < plan->optionCount; i++)
    {
        const double    RT = plan->optionData[i].R * plan->optionData[i].T;
        const double   sum = plan->h_CallValue[i].Expected;
        const double  sum2 = plan->h_CallValue[i].Confidence;
        const double pathN = plan->pathN;
        //Derive average from the total sum and discount by riskfree rate
        plan->callValue[i].Expected = (float)(exp(-RT) * sum / pathN);
        //Standard deviation
        double stdDev = sqrt((pathN * sum2 - sum * sum)/ (pathN * (pathN - 1)));
        //Confidence width; in 95% of all cases theoretical value lies within these borders
        plan->callValue[i].Confidence = (float)(exp(-RT) * 1.96 * stdDev / sqrt(pathN));

    }
*/
    for (int i = 0; i < plan->optionCount; i++)
         plan->callValue[i].Expected = plan->h_CallValue[i].Expected;
/*
    for (int i =0;i<1000;i++)
    {
      if (i%100 == 0){
      for (int j =0;j<4; j++)
        printf("%f ",plan->callValue[i*4+j].Expected);
      printf("\n");
    }
    }
*/
    checkCudaErrors(cudaFree(plan->rngStates));
    checkCudaErrors(cudaFreeHost(plan->h_CallValue));
    checkCudaErrors(cudaFreeHost(plan->h_OptionData));
    checkCudaErrors(cudaFree(plan->d_CallValue));
    checkCudaErrors(cudaFree(plan->d_OptionData));
}

//Main computations
extern "C" void MonteCarloGPU(TOptionPlan *plan, cudaStream_t stream)
{
    __TOptionValue *h_CallValue = plan->h_CallValue;

    if (plan->optionCount <= 0 || plan->optionCount > MAX_OPTIONS)
    {
        printf("MonteCarloGPU(): bad option count.\n");
        return;
    }


    __TOptionData * h_OptionData = (__TOptionData *)plan->h_OptionData;

    for (int i = 0; i < plan->optionCount; i++)
    {
        const double           T = plan->optionData[i].T;
        const double           R = plan->optionData[i].R;
        const double           V = plan->optionData[i].V;
        const double       MuByT = (R - 0.5 * V * V) * T;
        const double    VBySqrtT = V * sqrt(T);
        h_OptionData[i].S        = (real)plan->optionData[i].S;
        h_OptionData[i].X        = (real)plan->optionData[i].X;
        h_OptionData[i].MuByT    = (real)MuByT;
        h_OptionData[i].VBySqrtT = (real)VBySqrtT;
    }

    checkCudaErrors(cudaMemcpyAsync(
                        plan->d_OptionData,
                        h_OptionData,
                        plan->optionCount * sizeof(__TOptionData),
                        cudaMemcpyHostToDevice, stream
                    ));
/*
    MonteCarloOneBlockPerOption<<<plan->gridSize, THREAD_N, 0, stream>>>(
        plan->rngStates,
        (__TOptionData *)(plan->d_OptionData),
        (__TOptionValue *)(plan->d_CallValue),
        plan->pathN,
        plan->optionCount
    );
*/
    MonteCarloOneBlockPerOption<<<plan->optionCount/(256*4), 256, 0, stream>>>(
        plan->rngStates,
        (__TOptionData *)(plan->d_OptionData),
        (__TOptionValue *)(plan->d_CallValue),
        plan->pathN,
        plan->optionCount
    );
    getLastCudaError("MonteCarloOneBlockPerOption() execution failed\n");


    checkCudaErrors(cudaMemcpyAsync(
                        h_CallValue,
                        plan->d_CallValue,
                        plan->optionCount * sizeof(__TOptionValue), cudaMemcpyDeviceToHost, stream
                    ));

    cudaDeviceSynchronize();
/*
    for (int i=0; i<16;i++){
      for (int j =0; j<6; j++)
        printf("%f ", h_CallValue[i*16+j].Expected);

      printf("\n");
    }
*/

    //cudaDeviceSynchronize();
}
