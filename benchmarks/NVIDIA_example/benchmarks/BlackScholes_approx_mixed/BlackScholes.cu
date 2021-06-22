/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 sample number 0 input : 23.475470 69.922974 2.563801

 sample number 0  output : 0.099900

 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <stdio.h>
////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
);


////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 100000000;
const int  NUM_ITERATIONS = 10;


const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;


__constant__ half weight_1_half_d[48];
__constant__ half2 weight_2_half_d[8];

__constant__ half bias_1_half_d[8];

__constant__ half mean_d[6];
__constant__ half scale_d[6];




float weight_1 [48] = {0.827189, -0.00428317, 1.32327, 0.00471268, 0.403778, 0.00130504, 1.14673, 0.799463, -0.983907, 0.010704, -1.09407, -0.00301159, -2.18479, 0.00439583, -2.2148, -2.91757, -0.723392, 0.00141602, 0.995869, 0.00876862, -0.329325, -0.00879866, 0.230351, -0.164373, -0.00368561, 1.05699, -0.00107875, 0.494024, 0.00386155, 1.54069, 0.00487635, -0.00339686, -0.00416809, -2.61818, -2.77985e-05, -3.00977, -0.000440489, -1.42552, 0.00267766, 0.00182011, -0.00247033, 0.2442, 0.00312264, -0.298617, -9.05508e-05, 1.19926, 0.00262079, -0.00484339};
float bias_1 [8] = {-1.45582, -2.07027, 0.351997, -3.93356, -3.24358, 0.196985, -1.19395, -3.15805};
float weight_2 [16] = {-1.8426, -0.00188333, -0.00129177, 2.67614, 1.1904, -0.00190735, -0.00234156, 5.04987, 5.70659, -0.00810299, 0.00183688, 1.23903, 1.89747, 0.00389173, 3.34038, -0.00415401};
float bias_2[2] = {0.0829115, 0.114646};
float mean[6] = {17.49641715, 50.53246736,  5.12448171, 17.49819248, 50.48997263,  5.12594562};
float scale[6]=  {0.13856125, 0.03498948, 0.35525094, 0.13857643 ,0.03498012, 0.35534117};

half weight_1_half[48], bias_1_half[8], bias_2_half[2];
half mean_half[6], scale_half[6];
half2 weight_2_half[8];
void prepare_half_prec_weights(){
		for (int i =0;i < 48; i++)
			weight_1_half[i] = __float2half_rn(weight_1[i]);
		for (int i = 0; i< 8 ;i ++){
			weight_2_half[i] = __floats2half2_rn(weight_2[2*i],weight_2[2*i+1]);
			bias_1_half[i] = __float2half_rn(bias_1[i]);
		}
		for (int i = 0; i<2 ;i ++){

			bias_2_half[i] = __float2half_rn(bias_2[i]);
		}
    for (int i =0; i<6; i++){
      mean_half[i] = __float2half_rn(mean[i]);
      scale_half[i] = __float2half_rn(scale[i]);
    }


}

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Start logs
    printf("[%s] - Starting...\n", argv[0]);
    prepare_half_prec_weights();
    //'h_' prefix - CPU (host) memory space
    float
    //Results calculated by CPU for reference
    *h_CallResultCPU,
    *h_PutResultCPU,
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    half
    *h_CallResultGPU_half,
    *h_StockPrice_half,
    *h_OptionStrike_half,
    *h_OptionYears_half,
    *d_CallResult_half,
    *d_StockPrice_half,
    *d_OptionStrike_half,
    *d_OptionYears_half;

    double
    delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

    StopWatchInterface *hTimer = NULL;
    int i;

    findCudaDevice(argc, (const char **)argv);

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");
    h_CallResultCPU = (float *)malloc(OPT_SZ);
    h_PutResultCPU  = (float *)malloc(OPT_SZ);
    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);


    h_CallResultGPU_half = (half *)malloc(OPT_N*sizeof(half));
    h_StockPrice_half    = (half *)malloc(OPT_N*sizeof(half));
    h_OptionStrike_half  = (half *)malloc(OPT_N*sizeof(half));
    h_OptionYears_half   = (half *)malloc(OPT_N*sizeof(half));

    float * h_temp_result   = (float *)malloc(256 * sizeof(float));
    float * d_temp_result;
    cudaMalloc((void **)&d_temp_result, 256 * sizeof(float));
    int speed = 10; // min 0 max 10
    if (argc >1)
      speed =atoi(argv[1]);

    printf("running with speed %d \n", speed);
  //  float * d_weight1;
  //  cudaMalloc((void **)&d_weight1, 256 * sizeof(float));

    printf("...allocating GPU memory for options.\n");
    checkCudaErrors(cudaMalloc((void **)&d_CallResult,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult,    OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice,   OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears,  OPT_SZ));

    checkCudaErrors(cudaMalloc((void **)&d_CallResult_half,   OPT_N*sizeof(half)));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice_half,   OPT_N*sizeof(half)));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike_half, OPT_N*sizeof(half)));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears_half,  OPT_N*sizeof(half)));


    printf("...generating input data in CPU mem.\n");
    //srand(5347);
    srand(1234);
    //Generate options set
    for (i = 0; i < OPT_N; i++)
    {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);

        h_StockPrice_half[i] = __float2half_rn(h_StockPrice[i]);
        h_OptionStrike_half[i] = __float2half_rn(h_OptionStrike[i]);
        h_OptionYears_half[i]  = __float2half_rn(h_OptionYears[i] );
    }

    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_StockPrice_half,  h_StockPrice_half,   OPT_N*sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionStrike_half, h_OptionStrike_half,  OPT_N*sizeof(half), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OptionYears_half,  h_OptionYears_half,   OPT_N*sizeof(half), cudaMemcpyHostToDevice));
    printf("Data init done.\n\n");

//half data


	cudaMemcpyToSymbol(weight_1_half_d, &weight_1_half, 48 * sizeof(half));
	cudaMemcpyToSymbol(bias_1_half_d, &bias_1_half, 8 * sizeof(half));
	cudaMemcpyToSymbol(weight_2_half_d, &weight_2_half, 16 * sizeof(half));
  //  cudaMemcpyToSymbol(bias_3_half_d, &bias_3_half, 5 * sizeof(half));
  cudaMemcpyToSymbol(mean_d, &mean_half, 6 * sizeof(half));
  cudaMemcpyToSymbol(scale_d, &scale_half, 6 * sizeof(half));



    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (i = 0; i < NUM_ITERATIONS; i++)
    {
      BlackScholesGPU<<<DIV_UP((OPT_N), 256*2), 256/*480, 128*/>>>(
            d_CallResult_half,
            d_PutResult,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            d_StockPrice_half,
            d_OptionStrike_half,
            d_OptionYears_half,
          //  d_temp_result ,
            RISKFREE,
            VOLATILITY,
            OPT_N,
            speed
        );
        getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
    printf("Effective memory bandwidth: %f GB/s\n", ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
    printf("Gigaoptions per second    : %f     \n\n", ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

    printf("BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u options, NumDevsUsed = %u, Workgroup = %u\n",
           (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime*1e-3, (2 * OPT_N), 1, 256);


    printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
//    checkCudaErrors(cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_CallResultGPU_half, d_CallResult_half, OPT_N * sizeof(half), cudaMemcpyDeviceToHost));
    for (i = 0; i < OPT_N; i++)
      h_CallResultGPU[i] = __half2float(h_CallResultGPU_half[i]);
//    checkCudaErrors(cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost));
  //  checkCudaErrors(cudaMemcpy(h_temp_result,  d_temp_result,  256 * sizeof(float), cudaMemcpyDeviceToHost));
/*
    printf("printing output for testing  \n");
    for (int i =0 ; i < 16 ; i++){
      //if(i%16 ==0){
      for (int j =0 ; j < 8; j ++)
        printf("%.4f ", h_CallResultGPU[i*8+j]);
      printf("\n");
  //  }

    }
*/
    printf("Checking the results...\n");
    printf("...running CPU calculations.\n\n");
    //Calculate options values on CPU
    BlackScholesCPU(
        h_CallResultCPU,
        h_PutResultCPU,
        h_StockPrice,
        h_OptionStrike,
        h_OptionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N
    );
/**/
    printf("Comparing the results...\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;
    int max_index = 0;
    for (i = 0; i < OPT_N; i++)
    {
        ref   = h_CallResultCPU[i];
        delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

        if (delta > max_delta)
        {
            max_delta = delta;
            max_index = i;
        }

        sum_delta += delta;
        sum_ref   += fabs(ref);
    }

    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
  //  printf("Max absolute error: %E %d\n\n", max_delta, max_index);

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");
    checkCudaErrors(cudaFree(d_OptionYears));
    checkCudaErrors(cudaFree(d_OptionStrike));
    checkCudaErrors(cudaFree(d_StockPrice));
    checkCudaErrors(cudaFree(d_PutResult));
    checkCudaErrors(cudaFree(d_CallResult));

    printf("...releasing CPU memory.\n");
    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);
    free(h_PutResultCPU);
    free(h_CallResultCPU);
    sdkDeleteTimer(&hTimer);
    printf("Shutdown done.\n");

    printf("\n[BlackScholes] - Test Summary\n");

    // Calling cudaProfilerStop causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaProfilerStop());

    if (L1norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
