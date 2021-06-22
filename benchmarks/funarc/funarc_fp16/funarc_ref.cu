#include <iostream>
#include <math.h>
#include <stdio.h>
#include <cuda_fp16.h>
#define TYPE1 float
#define TYPE2 float
#define TYPE3 float
#define TYPE4 float
/*
 * 0.085340 ms
8.5531821794
abs err 2.75740586  rel err 0.47576126

 * */
/***
 * 100k
 * Elapsed time : 0.011452 ms
8.4106061017
abs err 2.61482978  rel err 0.45116127

 * 
 * */ 
 /**
  * 1m
  *  0.085744 ms
5.2551581033
abs err 0.54061822  rel err 0.09327796
  * 
  * */
  
  /**
   * 
   *  	0.011482 ms
		5.4107195504
		abs err 0.38505677  rel err 0.06643748
   * 
   * */
 
double fun_ref( double x){
  int k, n = 5;
  double t1;
  double d1 = 1.0;
  t1 = x;
  for ( k = 1; k <= n; k++ ){
      d1 = 2.0 * d1;
      t1 = t1+ sin(d1 * x)/d1;
    }
    return t1;
}
__global__ void fun_gpu(float x[], float y[], int nthreads, float h){
  //y = fun(x)
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int k, n = 5;
  half2 input = __floats2half2_rn(x[tid],x[tid+1]);
  if (tid < nthreads-1) {
    half2 t1;
    half2 d1 = __float2half2_rn(1.0);
    half2 x_temp = input;
    t1 = input;
    for ( k = 1; k <= n; k++ )
      {
        d1 = __float2half2_rn(2.0) * d1;
        half2 sin_res = h2sin(d1 * x_temp);
        t1 = t1 + sin_res/d1;
      }
      
    //float2 output = __half22float2(t1);
    half2 output = t1;
    y[tid] =  hsqrt(__float2half_rn(h*h) + (output.y - output.x) * (output.y - output.x));
  }
}


int main( int argc, char **argv) {
  int i,n = 1000000;
  double h, t1, t2, dppi;
  double s1;
  //cuda def
  cudaEvent_t start, stop;
  float elapsedTime;
  float *d_x, *d_y, *h_x, *h_y ;
  size_t size = n*sizeof(float);

  h_x = (float*) malloc(size);
  h_y = (float*) malloc(size);
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);


  t1 = -1.0;
  dppi = acos(t1);
  s1 = 0.0;
  t1 = 0.0;
  h = dppi / n;
  for ( i = 1; i <= n; i++){
    h_x[i-1] = i * h;
  }
    /* Copy vectors from host memory to device memory */
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

  int threads_per_block = 256;

  int block_count = (n + threads_per_block - 1)/threads_per_block;
  cudaEventCreate(&start);
  cudaEventRecord(start,0);
  for (int i =0;i < 100; i ++)
    fun_gpu<<<block_count, threads_per_block>>>(d_x, d_y, n, h);

  cudaDeviceSynchronize();
  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time : %f ms\n" ,elapsedTime/100.0);
  cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
  s1 = 0.000096; //the first loop
  for ( i = 1; i <= n; i++)
    {
     // t2 = h_y[i-1];
      s1 = s1 + h_y[i-1];// sqrt(h*h + (t2 - t1) * (t2 - t1));
    //  t1 = t2;
    }
  double ref_value = 5.7957763224;
  printf("%.10f\n",s1);
  printf("abs err %.8f  rel err %.8f\n", fabs(s1-ref_value), fabs((s1-ref_value)/ref_value) );
  return 0;
}
