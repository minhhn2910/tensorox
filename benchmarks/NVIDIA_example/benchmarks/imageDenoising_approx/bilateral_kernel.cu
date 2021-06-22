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
__device__ uint make_color(float r, float g, float b, float a)
{
    return
        ((int)(__saturatef(a) * 255.0f) << 24) |
        ((int)(__saturatef(b) * 255.0f) << 16) |
        ((int)(__saturatef(g) * 255.0f) <<  8) |
        ((int)(__saturatef(r) * 255.0f) <<  0)
        ;
}
//column pass using coalesced global memory reads
__global__ void
d_bilateral_filter(uint *od, int w, int h,
                   float e_d,  int r,  float* debug_output)
{
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
   od[y * w + x] = make_color(output.x, output.y,output.z,0);
  //    od[y * w + x] = rgbaFloatToInt(output);
}

extern "C"
void initTexture(int width, int height, uint *hImage)
{
    // copy image data to array
    checkCudaErrors(cudaMallocPitch(&dImage, &pitch, sizeof(uint)*width, height));
    checkCudaErrors(cudaMallocPitch(&dTemp,  &pitch, sizeof(uint)*width, height));
    checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint)*width,
                                 sizeof(uint)*width, height, cudaMemcpyHostToDevice));

   const float weight_1[256] = {0.0942194, -0.059401, -0.0178573, 0.0545111, 0.0392997, -0.157194, -0.0112189, 0.0851512, 0.0960393, -0.0679004, 0.0564275, 0.0570569, 0.0333527, 0.13094, -0.00273129, -0.0516536, 0.10171, -0.0420088, -0.040383, 0.0196547, -0.0505634, 0.0106072, 0.0610578, -0.00753733, 0.0779034, -0.0594784, -0.0144471, 0.0564601, 0.0277438, -0.148763, -0.0165365, 0.0844381, 0.0912005, -0.0720934, 0.0606311, 0.0543231, 0.0343399, 0.135044, -0.00334968, -0.0540848, 0.0961118, -0.0475447, -0.0386866, 0.0238709, -0.0572375, 0.0142017, 0.0633166, -0.0116256, 0.140185, -0.0503148, -0.0108386, 0.0501234, 0.0231175, -0.153156, -0.0120477, 0.0814653, 0.0827782, -0.0600503, 0.0605086, 0.0481634, 0.0289848, 0.132505, -0.000607221, -0.0528754, 0.0698749, -0.0377386, -0.0363277, 0.0266987, -0.0598342, 0.0101453, 0.0606295, -0.0118145, 0.0835931, -0.0590238, -0.0132593, 0.0557462, 0.0380344, -0.165798, -0.00795345, 0.0859377, 0.114278, -0.0728588, 0.0579844, 0.0713027, 0.0302509, 0.130784, -0.00424411, -0.0571617, 0.0481145, -0.0418375, -0.0403091, 0.0320959, -0.0586127, 0.017239, 0.0632288, -0.0141896, -0.47398, -0.351944, 0.0244336, 0.373026, 0.056476, -0.157452, 0.0595016, 0.437411, -0.566767, -0.376237, 0.420596, 0.415824, 0.0510919, 0.208966, 0.0685775, -0.277506, -0.503595, -0.311233, -0.385589, 0.359302, -0.103584, 0.0792294, 0.238188, -0.0852752, 0, 0, 0, 0, 0, 0, 0, 0, 0.101377, -0.0631646, -0.0106824, 0.053499, 0.032648, -0.159127, -0.00850739, 0.0853669, 0.0436172, -0.0769351, 0.0593629, 0.0556862, 0.0401097, 0.132232, -0.00620525, -0.0587907, 0.0854161, -0.045844, -0.0400369, 0.0368831, -0.0582402, 0.0147034, 0.0631251, -0.0150093, 0.0292751, -0.0436042, -0.0177396, 0.0323105, 0.0383469, -0.175741, -0.00530103, 0.0867386, 0.065484, -0.0523973, 0.0501531, 0.0481933, 0.02964, 0.123206, -0.0139933, -0.0553732, -0.0252474, -0.0279236, -0.0374022, 0.0128334, -0.0715334, 0.0173065, 0.0579969, -0.0145371, -0.00344323, -0.0533086, -0.0158497, 0.0300328, 0.0429003, -0.170118, -0.0128307, 0.0901911, -0.0360705, -0.0627324, 0.050902, 0.0499202, 0.0389883, 0.123856, -0.0242546, -0.0555833, -0.0368029, -0.0359352, -0.0413363, 0.0197517, -0.0650649, 0.00997995, 0.0611337, -0.0137108, 0.0233444, -0.0572051, -0.0164571, 0.0313132, 0.0478569, -0.17939, -0.0100068, 0.0861748, -0.000341296, -0.0655894, 0.0499444, 0.0447304, 0.0391042, 0.118343, -0.0222376, -0.0545189, 0.0136598, -0.0333381, -0.0450652, 0.0197653, -0.0548299, 0.00428894, 0.0521546, -0.012617     , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

   const float bias_1[8] = {-0.840858, -0.195456, 0.713454, 0.211434, 0.112341, -0.583816, 0.587517, 1.20987};
  const float weight_2 [24] = {
  -0.110958, -0.116568, -0.0949234, -0.102665, -0.116939, -0.0864574, 0.0280559, 0.0906449, -0.10507, 0.109485, 0.122846, 0.0930053, -0.00222155, 0.0369145, -0.0647695, 0.0423279, -0.0508375, -0.00427782, -0.0206118, 0.00657966, 0.103938, 0.163978, -0.0852381, -0.0560106
  };
     const float bias_2 [8] = {0.361678, 0.464349, 0.393955,0,0,0,0,0};

   const float means[28] = {0.594013, 0.453431, 0.318865, 0.593403, 0.452778, 0.318012, 0.592759, 0.452092, 0.317104, 0.595331, 0.454773, 0.3204, 0.594722, 0.454124, 0.319551, 0, 0.594077, 0.453441, 0.318644, 0.596562, 0.456032, 0.321863, 0.595955, 0.455388, 0.321018, 0.59531, 0.454709, 0.320114};
   const float scale [28] ={3.79913, 3.81843, 4.04111, 3.79833, 3.82209, 4.04848, 3.798, 3.82657, 4.05739, 3.79953, 3.80972, 4.0232, 3.79872, 3.81329, 4.0304, 1, 3.79836, 3.81768, 4.03913, 3.80122, 3.8027, 4.00792, 3.80039, 3.80616, 4.01494, 3.79999, 3.81045, 4.02349};
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
            dDest, width, height, e_d, radius,d_output);

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
