#include <iostream>
#include <math.h>
#include <stdio.h>
#define TYPE1 float
#define TYPE2 float
#define TYPE3 float
#define TYPE4 float
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

__constant__ half weight_1_half_d[32];
__constant__ half2 weight_2_half_d[8];

__constant__ half bias_1_half_d[8];
#define BIAS2 -8.16834

__device__ __inline__ half relu( half x){
  return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0) ;
}
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
	half2 bias_2_dev = __floats2half2_rn(-9.51932, -9.23336);
   int real_tid =  threadIdx.x;
    int warp_id = real_tid /32;
    int warp_lane = real_tid %32;

  //  __shared__ half A[8][512];
    __shared__ half weight_1_shared[128];
    __shared__ half bias_1_shared[256];
    __shared__ half neuron_out[8][512];
 
    if (real_tid <32){
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
			  neuron_out[warp_id][warp_lane+ 0*32] = x[2*tid];
			  neuron_out[warp_id][warp_lane+ 1*32] = x[2*tid+1];
			  neuron_out[warp_id][warp_lane+ 2*32] = x[2*tid+1];
			  neuron_out[warp_id][warp_lane+ 3*32] = x[2*tid+2];			
		// }
  		wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
  		wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared, 8);
  		wmma::load_matrix_sync(c_frag, (const half*)&bias_1_shared, 32, wmma::mem_col_major);

  		wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
  		for (int i = 0; i< c_frag.num_elements; i ++)
  			c_frag.x[i] = relu(c_frag.x[i]);
 		wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);

		//__syncwarp(); 
		half2 output1= __float2half2_rn(0.0);
		for (int i =0 ; i <8 ; i++ ) 
			output1 += __half2half2(neuron_out[warp_id][warp_lane+i*32])*weight_2_half_d[i];
		output1 += bias_2_dev;//__float2half_rn(BIAS2);

  if (tid*2 < nthreads-1) {
    y[tid*2] = __half2float(output1.x)*1e-6;
	y[tid*2+1] = __half2float(output1.y)*1e-6;

  }
}

float weight_1 [32] = {0, 0, 2.99042, 0.254513, 0, 0.354731, -0.59469, 0, 0, 0, 3.25762, 0.231921, 0, 0.750223, -1.11301, 0, 0, 0, 3.23843, 0.714279, 0, 0.57145, -0.560876, 0, 0, 0, 2.34076, 0.819863, 0, 0.0278206, -0.839322, 0};
float bias_1 [8] = {-0.0620128, -0.09748, -2.7673, 10.0375, -0.0233305, 9.25079, 8.90134, -2.03759e-05};
float weight_2 [16] = {0, 0, 0, 0, 4.36422, 4.36678, -4.56669, -4.69579, 0, 0, -3.4661, -3.34731, 11.7329, 11.7235, 0, 0};
float bias_2[2] = {-9.51932, -9.23336};
half weight_1_half[32], bias_1_half[8], bias_2_half[2];
half2 weight_2_half[8];
void prepare_half_prec_weights(){
		for (int i =0;i < 32; i++)	
			weight_1_half[i] = __float2half_rn(weight_1[i]);
		for (int i = 0; i< 8 ;i ++){
			weight_2_half[i] = __floats2half2_rn(weight_2[2*i],weight_2[2*i+1]);
			bias_1_half[i] = __float2half_rn(bias_1[i]);
		}
		for (int i = 0; i<2 ;i ++){

			bias_2_half[i] = __float2half_rn(bias_2[i]);
		}
		

}
	
int main( int argc, char **argv) {
  int i,n = 1000000;
  double h, t1, t2, dppi;
  prepare_half_prec_weights();
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


	cudaMemcpyToSymbol(weight_1_half_d, &weight_1_half, 32 * sizeof(half));
	cudaMemcpyToSymbol(bias_1_half_d, &bias_1_half, 8 * sizeof(half));
	cudaMemcpyToSymbol(weight_2_half_d, &weight_2_half, 16 * sizeof(half));

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

  int block_count = ((n + threads_per_block - 1)/threads_per_block)/2;
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
  
  for (int i = 0; i<16; i++){
	for (int j = 0; j < 16; j++)
		printf("%f ", h_y[i*16+j]);
	printf("\n");
	}
  
  printf("%.10f\n",s1);
  printf("abs err %.8f  rel err %.8f\n", fabs(s1-ref_value), fabs((s1-ref_value)/ref_value) );
  return 0;
}
