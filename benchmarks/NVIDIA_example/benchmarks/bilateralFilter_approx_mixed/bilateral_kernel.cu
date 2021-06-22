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

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
// The only dimensions currently supported by WMMA
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define SKEW_HALF 8 // from cuda sample avoid bank conflict

__constant__ half weight_1_half_d[256];
__constant__ half weight_2_half_d[24];

__constant__ half bias_1_half_d[8];
__constant__ half bias_2_half_d[8];

__constant__ float means_d[28];
__constant__ float scale_d[28];

__constant__ float cGaussian[64];   //gaussian array in device side
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;

uint *dImage  = NULL;   //original image
uint *dTemp   = NULL;   //temp array for iterations
size_t pitch;

/*
    Perform a simple bilateral filter.

    Bilateral filter is a nonlinear filter that is a mixture of range
    filter and domain filter, the previous one preserves crisp edges and
    the latter one filters noise. The intensity value at each pixel in
    an image is replaced by a weighted average of intensity values from
    nearby pixels.

    The weight factor is calculated by the product of domain filter
    component(using the gaussian distribution as a spatial distance) as
    well as range filter component(Euclidean distance between center pixel
    and the current neighbor pixel). Because this process is nonlinear,
    the sample just uses a simple pixel by pixel step.

    Texture fetches automatically clamp to edge of image. 1D gaussian array
    is mapped to a 1D texture instead of using shared memory, which may
    cause severe bank conflict.

    Threads are y-pass(column-pass), because the output is coalesced.

    Parameters
    od - pointer to output data in global memory
    d_f - pointer to the 1D gaussian array
    e_d - euclidean delta
    w  - image width
    h  - image height
    r  - filter radius
*/
__device__ __inline__ half relu( half x){
  return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0) ;
}

//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);
    return __expf(-mod / (2.f * d * d));
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

//column pass using coalesced global memory reads
__global__ void
d_bilateral_filter(uint *od, int w, int h,
                   float e_d,  int r,  int speed)
{

  int blockId = blockIdx.x + blockIdx.y * gridDim.x;;
  if(blockId%10 < speed){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;



    //debug
  //  if (blockIdx.x >1 || blockIdx.y  >1 )  return ;
  //  if (blockIdx.x >3 || blockIdx.y > 3)
  // return;
  //  int index = y*w +x;
  //  int index = x*h+y;
    int real_tid = threadIdx.y*blockDim.x + threadIdx.x;
    int warp_id = real_tid /32;
    int warp_lane = real_tid %32;

    //float sum = 0.0f;
    //float factor;
    float4 t = {0.f, 0.f, 0.f, 0.f};
    //float4 center = tex2D(rgbaTex, x, y);
    //no need to store center because it's included in the below loop
    //input record , fix r =1
    __shared__ half A[8][2][512];
    __shared__ half weight_1_shared[2][128];
    __shared__ half bias_1_shared[256];
  	__shared__ half weight_2_shared[128];
  	__shared__ half bias_2_shared[256];
    __shared__ half neuron_out[8][256];
  //  weight_1_shared[0][real_tid] = weight_1_half_d[real_tid];//__float2half_rn(0.0);
    if (real_tid <128){
      weight_1_shared[0][real_tid] = weight_1_half_d[real_tid];
      weight_1_shared[1][real_tid] = weight_1_half_d[128+real_tid];//__float2half_rn(0.0);
       weight_2_shared[real_tid] = 0.0;
    }
    //simple trick no need if/else, read as col_major later
    bias_1_shared[real_tid] = bias_1_half_d[warp_id];
    bias_2_shared[real_tid] = bias_2_half_d[warp_id];

    for (int i = 0; i<8 ; i++){
      A[i][0][real_tid] = 0.0;
      A[i][0][real_tid+256] = 0.0;
      A[i][1][real_tid] = 0.0;
      A[i][1][real_tid+256] = 0.0;
    }
/*
    if (real_tid <  8){
  		bias_1_shared[warp_lane_matrix] = bias_1_half_d[warp_lane_matrix];
  	} else if (real_tid <  16){
  		bias_2_shared[warp_lane_matrix] = bias_2_half_d[warp_lane_matrix];
  	}
*/
    __syncthreads();
    if (warp_lane<3){
      weight_2_shared[warp_lane+ warp_id*8] = weight_2_half_d[warp_lane+ warp_id*3];

    }
    //replicate


  //  __syncthreads();
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag_col;

    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    //  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    int count = 0 ;
    for (int i = -2; i <= 2; i+=2)
    {
        for (int j = -2; j <= 2; j+=2)
        {

            float4 curPix = tex2D(rgbaTex, x + j, y + i);
            if (count <= 4){
              A[warp_id][0][warp_lane+(count*3 + 0)*32] = (curPix.x - means_d[count*3 + 0])*scale_d[count*3 + 0];
              A[warp_id][0][warp_lane+(count*3 + 1)*32] = (curPix.y - means_d[count*3 + 1])*scale_d[count*3 + 1];
              A[warp_id][0][warp_lane+(count*3 + 2)*32] = (curPix.z - means_d[count*3 + 2])*scale_d[count*3 + 2];
            }
            else {
              A[warp_id][1][warp_lane+((count-5)*3 + 0)*32] = (curPix.x - means_d[count*3 + 1])*scale_d[count*3 + 1];
              A[warp_id][1][warp_lane+((count-5)*3 + 1)*32] = (curPix.y - means_d[count*3 + 2])*scale_d[count*3 + 2];
              A[warp_id][1][warp_lane+((count-5)*3 + 2)*32] = (curPix.z - means_d[count*3 + 3])*scale_d[count*3 + 3];

            }

            count ++ ;

          }
    }

    __syncthreads();

//    printf(" \n warpid %d warplane %d  %d count %d  %.4f %.4f", warp_id,warp_lane, warp_lane*16+(count-5)*3 + 0, count, __half2float(A[warp_id][warp_lane_loop][0][warp_lane*16+count*3 + 0]));
//   if (warp_lane == 15)
//    for (int k = 0; k < 256 ; k++)
//      debug_output[k] = weight_1_shared[0][k];


    float4 output;

    //loop to process 2 halves of data on warp cuz 1 tensor op only process 16 elems
//  int i =0;
  //layer 1

  		wmma::load_matrix_sync(a_frag_col, (const __half*)A[warp_id][0], 32);
  		wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared[0], 8);
  		wmma::load_matrix_sync(c_frag, (const half*)&bias_1_shared, 32, wmma::mem_col_major);
      //wmma::fill_fragment(acc_frag, 0.0f);
  		wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);

  		wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 8,wmma::mem_row_major);


  //2nd part

  		wmma::load_matrix_sync(a_frag_col, (const __half*)A[warp_id][1], 32);
  		wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared[1], 8);

  		wmma::load_matrix_sync(c_frag, (const half*)&neuron_out[warp_id], 8, wmma::mem_row_major);

  		wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
  		for (int i = 0; i< c_frag.num_elements; i ++)
  			c_frag.x[i] = relu(c_frag.x[i]);
__syncwarp();
  		wmma::store_matrix_sync((half*)A[warp_id][0], c_frag, 16,wmma::mem_row_major);

      //done 16x32 matrix mult 32x16 = 16x16 result, as simple as that :)


  //layer 2

  		wmma::load_matrix_sync(a_frag, (const __half*)A[warp_id][0], 16);
  //    wmma::load_matrix_sync(a_frag, (const __half*)A[row][0], 16);

  		wmma::load_matrix_sync(b_frag, (const __half*)&weight_2_shared, 8);
  		wmma::load_matrix_sync(c_frag, (const half*)&bias_2_shared, 32, wmma::mem_col_major);
    //  wmma::fill_fragment(acc_frag, 0.0f);
  		wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  		wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 8,wmma::mem_row_major);

      __syncwarp(); //for computing next half
      output.x =neuron_out[warp_id][warp_lane*8+0]; //indexing
      output.y =neuron_out[warp_id][warp_lane*8+1]; //indexing
      output.z =neuron_out[warp_id][warp_lane*8+2]; //indexing
      if (x >= w || y >= h)
      {
          return;
      }

  //    if(threadIdx.x == 15 &&  threadIdx.y ==15)
  //      printf("here %d %d %d %d\n", x ,y, gridDim.x,gridDim.y);
      od[y * w + x] = rgbaFloatToInt(output);

    }  else { //doing fp32

      int x = blockIdx.x*blockDim.x + threadIdx.x;
      int y = blockIdx.y*blockDim.y + threadIdx.y;

      if (x >= w || y >= h)
      {
          return;
      }

      float sum = 0.0f;
      float factor;
      float4 t = {0.f, 0.f, 0.f, 0.f};
      float4 center = tex2D(rgbaTex, x, y);

      for (int i = -r; i <= r; i++)
      {
          for (int j = -r; j <= r; j++)
          {
              float4 curPix = tex2D(rgbaTex, x + j, y + i);
              factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                       euclideanLen(curPix, center, e_d);             //range factor

              t += factor * curPix;
              sum += factor;
          }
      }

      od[y * w + x] = rgbaFloatToInt(t/sum);


      }
}

extern "C"
void initTexture(int width, int height, uint *hImage)
{
    // copy image data to array
    checkCudaErrors(cudaMallocPitch(&dImage, &pitch, sizeof(uint)*width, height));
    checkCudaErrors(cudaMallocPitch(&dTemp,  &pitch, sizeof(uint)*width, height));
    checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint)*width,
                                 sizeof(uint)*width, height, cudaMemcpyHostToDevice));

   const float weight_1[256] = {-0.0657791, -0.0201721, 0.0362567, 0.0333186, 0.0384123, 0.0530015, 0.110234, 0.0195397, 0.0991825, -0.0193234, 0.0378721, 0.0546284, -0.0330165, -0.103475, -0.170282, -0.0181621, -0.0666445, -0.0522058, -0.0313699, 0.146116, 0.00790881, 0.0776883, 0.0721288, -0.000736049, -0.0435574, 0.0796496, -0.0505595, 0.198745, 0.0477882, 0.0792191, 0.0425578, 0.0037292, -0.123985, -0.0412459, -0.0331504, 0.163446, -0.00431492, 0.0661788, 0.224878, 0.12859, 0.0579155, -0.217588, 0.141746, 0.0789089, -0.0231899, -0.140526, -0.190068, -0.127598, 0.0739053, -0.0184894, 0.0416795, -0.0155757, 0.0314131, 0.016547, -0.00459329, 0.00560788, 0.0374069, -0.0350406, 0.0205294, 0.0110671, -0.0122007, 0.0260977, -0.0987753, 0.0179266, -0.0278395, -0.0226546, -0.00293703, 0.121312, -0.00960341, -0.0402285, 0.0658931, -0.0262357, 0.108405, 0.0446989, -0.0339576, -0.0940991, 0.0268855, -0.20599, -0.278416, -0.0611665, -0.0942207, -0.0424281, -0.0532078, 0.136527, -0.0499914, -0.235894, 0.139554, 0.0279778, 0.0516251, -0.151551, 0.157577, 0.139216, 0.0125877, 0.175348, -0.061978, -0.0324943, -0.335809, -0.239873, 0.0621758, -0.154898, 0.433237, 0.220987, 0.598557, 0.108797, -0.517873, -0.465497, 0.424744, -0.330291, -0.123012, 0.185307, 0.269609, 0.436808, -0.487459, -0.571338, 0.56162, -0.567574, -0.042773, 0.220803, 0.455976, -0.356824, 0, 0, 0, 0, 0, 0, 0, 0, -0.0093808, 0.0573891, -0.0493341, 0.0815599, 0.0363781, -0.0401859, -0.131154, -0.0203684, 0.12635, -0.0969621, 0.0842884, -0.0257074, -0.0152736, -0.0179767, -0.0610924, 0.0830209, 0.112574, -0.167227, 0.113665, 0.0745161, 0.00859468, 0.0747864, -0.0910726, -0.05999, -0.0246147, 0.00171159, -0.0141572, -0.0183383, 0.00120302, -0.161673, -0.0636776, -0.0521884, 0.00365139, 0.0365182, 0.0907007, 0.0655018, 0.00704481, 0.170215, -0.0641004, 0.0825097, 0.119897, -0.0938504, -0.034036, 0.000810103, -0.0269976, -0.218659, -0.054387, -0.08475, 0.226661, 0.0328325, 0.00468056, -0.0958401, 0.0665298, 0.116667, -0.148665, 0.0509406, 0.285547, -0.196717, 0.0229095, -0.0337319, -0.0239877, -0.131938, -0.0612404, 0.0321878, -0.00144399, -0.106171, 0.145874, -0.0130349, -0.02173, -0.0746803, -0.195182, -0.102325, 0.0463445, -0.0455709, 0.0917308, 0.107777, 0.0279806, -0.0633389, -0.0480169, -0.0314363, 0.151845, -0.0401172, 0.0474296, -0.0743281, -0.00871706, 0.0459901, -0.185976, 0.0454504, 0.214521, -0.128203, 0.0280383, -0.0481311, 0.0250123, 0.136998, -0.0685455, 0.013213,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

   const float bias_1[8] = {-0.315055, 0.11735, -0.0777091, -0.382569, 1.07296, 0.38151, -0.272254, 0.547409};
  const float weight_2 [24] = {-0.0567982, -0.062898, -0.0658947, -0.0548249, -0.104245, -0.09974, 0.0755432, 0.145047, 0.140376, -0.0414028, -0.0529998, -0.057597, 0.355631, -0.0236014, -0.0141915, -0.0785561, -0.0687004, 0.00966274, 0.0806411, 0.0877628, 0.0754708, 0.0536888, 0.186424, -0.11164};
     const float bias_2 [8] = {0.221467, 0.413276, 0.39817,0,0,0,0,0};

   const float means[28] = {0.596071, 0.45235, 0.315686, 0.595453, 0.451718, 0.314807, 0.594786, 0.451037, 0.313879, 0.597391, 0.45368, 0.317216, 0.596774, 0.453049, 0.316339, 0, 0.596107, 0.452367, 0.31541, 0.598641, 0.454941, 0.318676, 0.598025, 0.45431, 0.317799, 0.597357, 0.453628, 0.316872};
   const float scale [28] ={3.85664, 3.93223, 4.15975, 3.85571, 3.93611, 4.16819, 3.85532, 3.94101, 4.17831, 3.85721, 3.92249, 4.14006, 3.85624, 3.92631, 4.14832, 1, 3.85581, 3.93113, 4.15823, 3.85904, 3.91466, 4.12339, 3.85804, 3.91841, 4.13149, 3.85757, 3.92316, 4.14122};
   half weight_1_half[256],bias_1_half[16],weight_2_half[48], bias_2_half[16];
   for (int i=0;i < 256; i++)
     weight_1_half[i] = __float2half_rn(weight_1[i]);

   for (int i=0;i < 24; i++)
     weight_2_half[i] = __float2half_rn(weight_2[i]);

   for (int i=0;i < 8; i++){
     bias_1_half[i] = __float2half_rn(bias_1[i]);
     bias_2_half[i] = __float2half_rn(bias_2[i]);
   }
   cudaMemcpyToSymbol(weight_1_half_d, &weight_1_half, 256 * sizeof(half));
   cudaMemcpyToSymbol(weight_2_half_d, &weight_2_half, 24 * sizeof(half));
   cudaMemcpyToSymbol(bias_1_half_d, &bias_1_half, 8 * sizeof(half));
   cudaMemcpyToSymbol(bias_2_half_d, &bias_2_half, 8 * sizeof(half));

   cudaMemcpyToSymbol(means_d,&means, 28 * sizeof(float));
   cudaMemcpyToSymbol(scale_d, &scale, 28 * sizeof(float));



}

extern "C"
void freeTextures()
{
    checkCudaErrors(cudaFree(dImage));
    checkCudaErrors(cudaFree(dTemp));
}

/*
    Because a 2D gaussian mask is symmetry in row and column,
    here only generate a 1D mask, and use the product by row
    and column index later.

    1D gaussian distribution :
        g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

    parameters:
    og - output gaussian array in global memory
    delta - the 2nd parameter 'd' in the above function
    radius - half of the filter size
             (total filter size = 2 * radius + 1)
*/
extern "C"
void updateGaussian(float delta, int radius)
{
    float  fGaussian[64];

    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta));
    }

    checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1)));
}

/*
    Perform 2D bilateral filter on image using CUDA

    Parameters:
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    e_d    - euclidean delta
    radius - filter radius
    iterations - number of iterations
*/

// RGBA version
extern "C"
double bilateralFilterRGBA(uint *dDest,
                           int width, int height,
                           float e_d, int radius, int iterations,
                           StopWatchInterface *timer)
{
    // var for kernel computation timing
    double dKernelTime;

#define TRAIN_DATA
      float * h_input, *h_output;
      float* d_input, *d_output;

      int output_size = width*height*3;
      int speed = 9;
/*
      h_output = (float*) malloc(output_size*sizeof(float));

      cudaMalloc((void**) &d_output, output_size*sizeof(float));
*/
    // Bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dImage, desc, width, height, pitch));

    for (int i=0; i<iterations; i++)
    {
        // sync host and start kernel computation timer
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_bilateral_filter<<< gridSize, blockSize>>>(
            dDest, width, height, e_d, radius,speed);

        // sync host and stop computation timer
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width,
                                         sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
        }
    }
/*
    cudaMemcpy( h_output, d_output, output_size*sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n");
    for (int i =0 ; i<32 ; i ++){
      for (int j =0 ; j<8; j ++)
        printf("%.4f ", h_output[i*8+j]);
      printf("\n");
    }
*/
    return ((dKernelTime/1000.)/(double)iterations);
}
