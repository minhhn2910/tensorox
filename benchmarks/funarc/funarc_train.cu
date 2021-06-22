#include <iostream>
#include <math.h>
#include <stdio.h>
#define TYPE1 double
#define TYPE2 double
#define TYPE3 double
#define TYPE4 double
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
__global__ void fun_gpu(double x[], double y[], int nthreads, double h){
  //y = fun(x)
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int k, n = 5;
  if (tid < nthreads-1) {
    double t1;
    double d1 = 1.0;
    double x_temp = x[tid];
    t1 = x[tid];
    for ( k = 1; k <= n; k++ )
      {
        d1 = 2.0 * d1;
        TYPE4 sin_res = sin(d1 * x_temp);
        t1 = t1 + sin_res/d1;
      }
      
    double t2;
    double d2 = 1.0;
    double x_temp2 = x[tid+1];
    t2 = x[tid+1];
    for ( k = 1; k <= n; k++ )
      {
        d2 = 2.0 * d2;
        TYPE4 sin_res = sin(d2 * x_temp2);
        t2 = t2 + sin_res/d2;
      }
      
    y[tid] =  sqrt(h*h + (t2 - t1) * (t2 - t1));
  }
}


int main( int argc, char **argv) {
  int i,n = 1000000;
  double h, t1, t2, dppi;
  double s1;
  //cuda def
  cudaEvent_t start, stop;
  float elapsedTime;
  double *d_x, *d_y, *h_x, *h_y ;
  size_t size = n*sizeof(double);

  h_x = (double*) malloc(size);
  h_y = (double*) malloc(size);
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
  for (int i =0;i < 10; i ++)
    fun_gpu<<<block_count, threads_per_block>>>(d_x, d_y, n, h);

  cudaDeviceSynchronize();
  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time : %f ms\n" ,elapsedTime);
  cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
  s1 = 0.000096; //the first loop
  for ( i = 1; i <= n; i++)
    {
     // t2 = h_y[i-1];
      s1 = s1 + h_y[i-1];// sqrt(h*h + (t2 - t1) * (t2 - t1));
    //  t1 = t2;
    }
  double ref_value = 5.7957763224;
  
#define TRAIN_DATA
#ifdef TRAIN_DATA
{
  FILE *fp;
  fp = fopen("input.txt", "w");
  fprintf(fp, "%.8e,%.8e", h_x[0],h_x[1]);
  for (int i=1; i < n-1; i++ ){
	fprintf(fp, ",%.8e,%.8e", h_x[i], h_x[i+1]);   
  }
  fclose(fp);
}

{
  FILE *fp;
  fp = fopen("output.txt", "w");
  fprintf(fp, "%.8e", h_y[0]);
  for (int i=1; i <n-1; i++ ){
    fprintf(fp, ",%.8e", h_y[i]);
  }
  fclose(fp);
}
#endif 

  printf("%.10f\n",s1);
  printf("abs err %.8f  rel err %.8f\n", fabs(s1-ref_value), fabs((s1-ref_value)/ref_value) );
  return 0;
}
