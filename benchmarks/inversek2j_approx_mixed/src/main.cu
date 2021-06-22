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

#include <cuda_fp16.h>
//#define MAX_LOOP 1000
//#define MAX_DIFF 0.15f
//#define NUM_JOINTS 3



#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;


#define MAX_LOOP 1000
#define MAX_DIFF 0.15f
#define NUM_JOINTS 3
#define PI 3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)


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
__constant__ half weight_2_half_d[64];
__constant__ half weight_3_half_d[64];
__constant__ half bias_1_half_d[8];
__constant__ half bias_2_half_d[8];
__constant__ half bias_3_half_d[8];
__constant__ float mean_d[4];
__constant__ float scale_d[4];
__device__ __inline__ half relu( half x){
  return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0) ;
}
__global__ void invkin_kernel(float *xTarget_in, float *yTarget_in, float *angles, int size, float err_thresh, int speed)
{
if (blockIdx.x %10 < speed){
	//tensor approx
	if(blockDim.x != 256) {
		printf("not supported block dimension , it must be 256\n");
		return;
	}
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int idx = tid;

   int real_tid =  threadIdx.x;
   int warp_id = real_tid /32;
   int warp_lane = real_tid %32;
   __shared__ half neuron_out[8][512];
   __shared__ half weight_1_shared[128];
   __shared__ half bias_1_shared[256];

   __shared__ half weight_2_shared[128];
   __shared__ half bias_2_shared[256];
   __shared__ half weight_3_shared[128];
   __shared__ half bias_3_shared[256];

   weight_1_shared[real_tid] = 0.0;
   weight_2_shared[real_tid] = 0.0;
   weight_3_shared[real_tid] = 0.0;
   for (int i = 0; i<8 ; i++){
     neuron_out[i][real_tid] = 0.0;
     neuron_out[i][real_tid+256] = 0.0;
   }
   __syncthreads();
  if (real_tid <32){
    weight_1_shared[real_tid] = weight_1_half_d[real_tid];
  }
  if (real_tid < 64){
    weight_2_shared[real_tid] = weight_2_half_d[real_tid];;
    weight_3_shared[real_tid] = weight_3_half_d[real_tid];;
  }

  //simple trick no need if/else, read as col_major later
  bias_1_shared[real_tid] = bias_1_half_d[warp_id];
  bias_2_shared[real_tid] = bias_2_half_d[warp_id];
  bias_3_shared[real_tid] = bias_3_half_d[warp_id];

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag_col;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

        neuron_out[warp_id][warp_lane+ 0*32] = (xTarget_in[2*tid] - mean_d[0])*scale_d[0];
        neuron_out[warp_id][warp_lane+ 1*32] = (yTarget_in[2*tid] - mean_d[1])*scale_d[1];
 			  neuron_out[warp_id][warp_lane+ 2*32] = (xTarget_in[2*tid+1] - mean_d[2])*scale_d[2];
 			  neuron_out[warp_id][warp_lane+ 3*32] = (yTarget_in[2*tid+1] - mean_d[3])*scale_d[3];

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


  wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
  wmma::load_matrix_sync(b_frag, (const __half*)&weight_3_shared, 8);
  wmma::load_matrix_sync(c_frag, (const half*)&bias_3_shared, 32, wmma::mem_col_major);

  wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
  wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);

  //__syncwarp();
  angles[(2*tid)*3 + 0 ] = __half2float(neuron_out[warp_id][warp_lane+0*32])*100.0;
  angles[(2*tid)*3 + 1 ] = __half2float(neuron_out[warp_id][warp_lane+1*32])*100.0;
  angles[(2*tid)*3 + 2 ] = __half2float(neuron_out[warp_id][warp_lane+2*32])*100.0;
  angles[(2*tid+1)*3 + 0 ] = __half2float(neuron_out[warp_id][warp_lane+3*32])*100.0;
  angles[(2*tid+1)*3 + 1 ] = __half2float(neuron_out[warp_id][warp_lane+4*32])*100.0;
  angles[(2*tid+1)*3 + 2 ] = __half2float(neuron_out[warp_id][warp_lane+5*32])*100.0;

}else {
 {
  //doing float compute
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int temp_idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
 int idx = temp_idx*2;

      float angle_out[NUM_JOINTS];

      for(int i = 0; i < NUM_JOINTS; i++)
      {
        angle_out[i] = 0.0;
      }

      float max_err 	= err_thresh * (float)(NUM_JOINTS);
      float err 		= max_err + 1.f; // initialize error to something greater than error threshold

    // Initialize x and y data
    float xData[NUM_JOINTS_P1];
    float yData[NUM_JOINTS_P1];

    for (int i = 0 ; i < NUM_JOINTS_P1; i++)
    {
      xData[i] = i;
      yData[i] = 0.f;
    }

    for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
    {
      for (int iter = NUM_JOINTS; iter > 0; iter--)
      {
        float pe_x = xData[NUM_JOINTS];
        float pe_y = yData[NUM_JOINTS];
        float pc_x = xData[iter-1];
        float pc_y = yData[iter-1];
        float diff_pe_pc_x = pe_x - pc_x;
        float diff_pe_pc_y = pe_y - pc_y;
        float diff_tgt_pc_x = xTarget_in[idx] - pc_x;
        float diff_tgt_pc_y = yTarget_in[idx] - pc_y;
        float len_diff_pe_pc = sqrt(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
        float len_diff_tgt_pc = sqrt(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
        float a_x = diff_pe_pc_x / len_diff_pe_pc;
        float a_y = diff_pe_pc_y / len_diff_pe_pc;
        float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
        float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
        float a_dot_b = a_x * b_x + a_y * b_y;
        if (a_dot_b > 1.f)
          a_dot_b = 1.f;
        else if (a_dot_b < -1.f)
          a_dot_b = -1.f;
        float angle = acos(a_dot_b) * (180.f / PI);
        // Determine angle direction
        float direction = a_x * b_y - a_y * b_x;
        if (direction < 0.f)
          angle = -angle;
        // Make the result look more natural (these checks may be omitted)
        // if (angle > 30.f)
        // 	angle = 30.f;
        // else if (angle < -30.f)
        // 	angle = -30.f;
        // Save angle
        angle_out[iter - 1] = angle;
        for (int i = 0; i < NUM_JOINTS; i++)
        {
          if(i < NUM_JOINTS - 1)
          {
            angle_out[i+1] += angle_out[i];
          }
        }
      }
    }

    angles[idx * NUM_JOINTS + 0] = angle_out[0];
    angles[idx * NUM_JOINTS + 1] = angle_out[1];
    angles[idx * NUM_JOINTS + 2] = angle_out[2];

  } // doing 2 of this

{
 //doing float compute
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int temp_idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
int idx = temp_idx*2+1;

     float angle_out[NUM_JOINTS];

     for(int i = 0; i < NUM_JOINTS; i++)
     {
       angle_out[i] = 0.0;
     }

     float max_err 	= err_thresh * (float)(NUM_JOINTS);
     float err 		= max_err + 1.f; // initialize error to something greater than error threshold

   // Initialize x and y data
   float xData[NUM_JOINTS_P1];
   float yData[NUM_JOINTS_P1];

   for (int i = 0 ; i < NUM_JOINTS_P1; i++)
   {
     xData[i] = i;
     yData[i] = 0.f;
   }

   for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
   {
     for (int iter = NUM_JOINTS; iter > 0; iter--)
     {
       float pe_x = xData[NUM_JOINTS];
       float pe_y = yData[NUM_JOINTS];
       float pc_x = xData[iter-1];
       float pc_y = yData[iter-1];
       float diff_pe_pc_x = pe_x - pc_x;
       float diff_pe_pc_y = pe_y - pc_y;
       float diff_tgt_pc_x = xTarget_in[idx] - pc_x;
       float diff_tgt_pc_y = yTarget_in[idx] - pc_y;
       float len_diff_pe_pc = sqrt(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
       float len_diff_tgt_pc = sqrt(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
       float a_x = diff_pe_pc_x / len_diff_pe_pc;
       float a_y = diff_pe_pc_y / len_diff_pe_pc;
       float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
       float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
       float a_dot_b = a_x * b_x + a_y * b_y;
       if (a_dot_b > 1.f)
         a_dot_b = 1.f;
       else if (a_dot_b < -1.f)
         a_dot_b = -1.f;
       float angle = acos(a_dot_b) * (180.f / PI);
       // Determine angle direction
       float direction = a_x * b_y - a_y * b_x;
       if (direction < 0.f)
         angle = -angle;
       // Make the result look more natural (these checks may be omitted)
       // if (angle > 30.f)
       // 	angle = 30.f;
       // else if (angle < -30.f)
       // 	angle = -30.f;
       // Save angle
       angle_out[iter - 1] = angle;
       for (int i = 0; i < NUM_JOINTS; i++)
       {
         if(i < NUM_JOINTS - 1)
         {
           angle_out[i+1] += angle_out[i];
         }
       }
     }
   }

   angles[idx * NUM_JOINTS + 0] = angle_out[0];
   angles[idx * NUM_JOINTS + 1] = angle_out[1];
   angles[idx * NUM_JOINTS + 2] = angle_out[2];

 }
 // doing 2 of this

}//end else

}

float weight_1[32]= {-0.000115723, 0.0405895, 0.00150536, 0.00486455, -0.9502, 0.654486, -0.0136366, 1.35293, 0.0018889, -0.108206, 0.0026806, -0.00578669, -0.578652, 0.705829, 0.037353, 0.138391, -1.51673, -0.247739, 1.59409, 0.360832, -0.00188272, -0.00380421, 0.502039, 0.00188776, -0.402409, 0.175721, -0.354387, 0.621299, 0.00123498, 0.000636481, -0.356029, 0.000677776};
float bias_1[8] = {0.852395, 1.37688, 0.165074, 0.583182, -0.0139461, 0.861103, 1.53502, 0.918966};
float weight_2[64] = {0.202584, 0.328301, -0.00134902, 0, 0.714236, 0.579672, 1.35258, -0.00382785, 0.786266, 0.533189, -0.162781, 0, 1.54451, 0.477951, -0.574619, -0.587302, 1.02665, -0.754968, 0.00465825, 0, -0.50198, -1.0119, -0.678385, 0.00107931, 0.167012, -0.159305, -0.00354383, 0, 0.983958, 0.191672, 0.863519, -0.00491045, 0.9714, 0.712408, 0.0857966, 0, -0.118303, -0.0685089, -0.235389, 0, 0.279924, 0.250664, -1.19851, 0, 0.123626, -0.0587833, -0.126924, -0.204036, 0.692083, 1.05158, -0.0801055, 0, 0.520167, 0.855179, 0.867944, -0.290132, -0.0904754, -0.610555, 1.2727, 0, -0.567912, -0.128276, 0.910889, 0.670079};
float bias_2[8] = {0.45518, 0.722925, -0.266987, -0.0617215, 0.55139, 1.26111, 0.651831, -0.0267784};
float weight_3 [48] = {-0.051853, 0.307628, 1.27661, -0.0109751, -0.112567, -0.204493, 0.466815, 0.465612, 0.943865, 0.0995894, 0.332187, 0.752312, -0.207995, -0.727746, -2.16582, 0.00368551, 0.0233835, -0.0162442, 0, 0, 0, 0, 0, 0, 0.182312, 0.327259, 0.926951, 0.120316, 0.199394, 0.387562, -0.496074, 0.0336437, 1.02593, 0.0164394, 0.145168, 1.18152, 0.0150528, -0.344478, -1.34943, 0.137851, 0.343412, 0.897156, 0.571383, 1.13275, 2.63659, -0.0301548, -0.0431947, 0.00362586};
float bias_3 [8] = {0.640118, 0.389619, 0.812074, -0.338533, 0.106341, 0.863131};
half weight_1_half[32], weight_2_half[64], weight_3_half[64], bias_1_half[8], bias_2_half[8], bias_3_half[8];

void prepare_half_prec_weights(){
	for (int i =0 ; i< 8 ; i ++){
		bias_1_half[i] = __float2half_rn(bias_1[i]);
		bias_2_half[i] = __float2half_rn(bias_2[i]);
		bias_3_half[i] = __float2half_rn(bias_3[i]);
	}

	for (int i =0 ; i<32; i ++)
		weight_1_half[i] = __float2half_rn(weight_1[i]);

	for (int i = 0 ; i<64; i++)
		weight_2_half[i] = __float2half_rn(weight_2[i]);

    for (int i =0; i <8; i ++){
          for (int j =0 ; j <6 ; j ++)
            weight_3_half[i*8+j] = __float2half_rn(weight_3[i*6 + j]);
          for (int j = 6; j <8 ;j++)
            weight_3_half[i*8+j] = __float2half_rn(0.0);
    }
}

using namespace std;
int main(int argc, char* argv[])
{
	if(argc < 4)
	{
		std::cerr << "Usage: ./invkin.out <input file coefficients> <output file> <error threshold>" << std::endl;
		exit(EXIT_FAILURE);
	}
  int speed = 10;
  if (argc >4 )
    speed = atoi(argv[4]);
  printf("running at speed %d \n", speed);

	float* xTarget_in_h;
	float* yTarget_in_h;
	float* angle_out_h;

	prepare_half_prec_weights();

	cudaError_t cudaStatus;

	int data_size = 0;

	// process the files
	ifstream coordinate_in_file (argv[1]);
	ofstream angle_out_file (argv[2]);
	float err_thresh = atof(argv[3]);


	if(coordinate_in_file.is_open())
	{
		coordinate_in_file >> data_size;
		std::cout << "# Data Size = " << data_size << std::endl;
	}

	// allocate the memory
	xTarget_in_h = new (nothrow) float[data_size];
	if(xTarget_in_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	yTarget_in_h = new (nothrow) float[data_size];
	if(yTarget_in_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	angle_out_h = new (nothrow) float[data_size*NUM_JOINTS];
	if(angle_out_h == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}


	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

//scaler parameter
float mean[4] = {-4.7858e-04,  1.4443e+00 , -1.3808e-03 , 1.4450e+00};
float scale[4] = {0.8009, 1.4939, 0.8002, 1.4952};
//

	float* xTarget_in_h_ref = new (nothrow) float[data_size];
	float* yTarget_in_h_ref = new (nothrow) float[data_size];
	// add data to the arrays
	float xTarget_tmp, yTarget_tmp;
	int coeff_index = 0;
	while(coeff_index < data_size)
	{
		coordinate_in_file >> xTarget_tmp >> yTarget_tmp;

		for(int i = 0; i < NUM_JOINTS ; i++)
		{
			angle_out_h[coeff_index * NUM_JOINTS + i] = 0.0;
		}
		//remove if neccessary
		xTarget_in_h_ref[coeff_index] = xTarget_tmp;
		yTarget_in_h_ref[coeff_index] = yTarget_tmp;
		//doing scaling
/*		int indx = coeff_index%2;
		xTarget_tmp = (xTarget_tmp - means[indx*2])/scale[indx*2];
		yTarget_tmp = (yTarget_tmp - means[indx*2+1])/scale[indx*2+1];
*/
//		if(coeff_index < 10)
//			printf(" %f %f \n ",xTarget_tmp, yTarget_tmp);
		xTarget_in_h[coeff_index] = xTarget_tmp;
		yTarget_in_h[coeff_index++] = yTarget_tmp;

	}



	std::cout << "# Coordinates are read from file..." << std::endl;

	// memory allocations on the host
	float 	*xTarget_in_d,
			*yTarget_in_d;
	float 	*angle_out_d;

	cudaMalloc((void**) &xTarget_in_d, data_size * sizeof(float));
	cudaMalloc((void**) &yTarget_in_d, data_size * sizeof(float));
	cudaMalloc((void**) &angle_out_d,  data_size * NUM_JOINTS * sizeof(float));

	std::cout << "# Memory allocation on GPU is done..." << std::endl;

	cudaMemcpy(xTarget_in_d, xTarget_in_h, data_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yTarget_in_d, yTarget_in_h, data_size * sizeof(float), cudaMemcpyHostToDevice);


	cudaMemcpyToSymbol(weight_1_half_d, &weight_1_half, 32 * sizeof(half));
	cudaMemcpyToSymbol(weight_2_half_d, &weight_2_half, 64 * sizeof(half));
	cudaMemcpyToSymbol(weight_3_half_d, &weight_3_half, 64 * sizeof(half));

	cudaMemcpyToSymbol(bias_1_half_d, &bias_1_half, 8 * sizeof(half));
	cudaMemcpyToSymbol(bias_2_half_d, &bias_2_half, 8 * sizeof(half));
	cudaMemcpyToSymbol(bias_3_half_d, &bias_3_half, 8 * sizeof(half));

  cudaMemcpyToSymbol(mean_d, &mean, 4 * sizeof(float));
	cudaMemcpyToSymbol(scale_d, &scale, 4 * sizeof(float));



	std::cout << "# Data are transfered to GPU..." << std::endl;

	dim3 dimBlock	( 256, 1 );
	dim3 dimGrid	( data_size / (256*2), 1 ); //256x5 for each block
//	dim3 dimGrid	( 2, 1 ); //256x5 for each block


	cudaEventRecord(start, 0);

#pragma parrot.start("invkin_kernel")

	invkin_kernel<<<dimGrid, dimBlock>>>(xTarget_in_d, yTarget_in_d, angle_out_d, data_size, err_thresh, speed);

#pragma parrot.end("invkin_kernel")

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

	std::cout << "# Elapsed Time in `nrpoly3` kernel = " << elapsedTime << std::endl;
	std::cout << "# GPU computation is done ..." << std::endl;

	cudaMemcpy(angle_out_h, angle_out_d, data_size * NUM_JOINTS * sizeof(float), cudaMemcpyDeviceToHost);
/*
	printf("test result \n");
	for(int i = 0; i < 16; i++){
		//if (i %16 == 0){
			for (int j =0; j<6; j++)
				printf("%f ",angle_out_h[i*6+j]);
			printf("\n\n");
	//	}
	}

	{
		FILE *fp;
		fp = fopen("mlp_invk2j.txt", "w");
		for (int i = 0; i < data_size; i++)
		{
			 if (i == data_size -1 )
					fprintf(fp, "%f,%f,%f",angle_out_h[i*3],angle_out_h[i*3+1],angle_out_h[i*3+2] );
			 else
					fprintf(fp, "%f,%f,%f,",angle_out_h[i*3],angle_out_h[i*3+1],angle_out_h[i*3+2]);
		}
		fclose(fp);

	}
	*/
	for(int i = 0; i < data_size; i++)
	{
/*		int indx = coeff_index%5;
		xTarget_tmp = (xTarget_in_h[i]*scale[indx*2] + means[indx*2]);
		yTarget_tmp = (yTarget_in_h[i]*scale[indx*2+1] + means[indx*2+1]);
*/
		angle_out_file << xTarget_in_h_ref[i] << " " << yTarget_in_h_ref[i] << " ";
		//this is input => dont store
		for(int j = 0 ; j < NUM_JOINTS; j++)
		{
			angle_out_file << angle_out_h[i * NUM_JOINTS + j] << " ";
		}
		angle_out_file << std::endl;
	}

	// close files
	coordinate_in_file.close();
	angle_out_file.close();

	// de-allocate the memory
	delete[] xTarget_in_h;
	delete[] yTarget_in_h;
	delete[] angle_out_h;

	// de-allocate cuda memory
	cudaFree(xTarget_in_d);
	cudaFree(yTarget_in_d);
	cudaFree(angle_out_d);

	std::cout << "Thank you..." << std::endl;
}
