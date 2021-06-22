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


#define MAX_LOOP 1000
#define MAX_DIFF 0.15f
#define NUM_JOINTS 3
#define PI 3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)


//begin tensorox_define
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
__device__ __inline__ half relu( half x){
	 return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0);
}
__constant__ half weight_1_half_d[32];
__constant__ half bias_1_half_d[8];
__constant__ half weight_2_half_d[64];
__constant__ half bias_2_half_d[8];
__constant__ half weight_3_half_d[64];
__constant__ half bias_3_half_d[8];
float weight_1[32]= { -0.00011572289659901992,0.04058952989942363,0.0015053599456909836,0.004864553656876592,-0.950199594498499,0.6544858026396745,-0.013636600111152596,1.3529270698915818,0.001888896810332068,-0.10820555088839402,0.0026805953825300285,-0.005786686518018669,-0.5786515285252845,0.7058289165416208,0.03735300132257165,0.13839067200280458,-1.51672930569754,-0.2477388460726752,1.5940933936995243,0.36083220329852556,-0.0018827208389212986,-0.003804206496879246,0.5020393671233125,0.0018877632269713343,-0.4024088271991091,0.17572137141564056,-0.35438728594047614,0.621299455709366,0.00123498156587144,0.0006364809628087192,-0.3560293773544104,0.0006777761674820835 };
float bias_1[8]= { 0.8523948208931647,1.3768812056400659,0.16507405502030162,0.5831817780844972,-0.013946104633911341,0.8611034691188276,1.5350171792041536,0.9189664296971597 };
float weight_2[64]= { 0.2025844896146973,0.32830095209599797,-0.0013490185943327748,0.0,0.7142360395517193,0.5796715555863106,1.3525839572953409,-0.0038278500891424827,0.7862664886967025,0.5331891518850409,-0.16278116776268015,0.0,1.5445089568533954,0.4779506465101805,-0.5746185570952375,-0.5873023949781635,1.0266456099195973,-0.7549675695962053,0.004658252049854064,0.0,-0.5019803543127834,-1.0119040470925804,-0.6783845884804952,0.0010793084122511097,0.16701207813432947,-0.15930450562792742,-0.003543833354901427,0.0,0.9839584003561566,0.19167220169911775,0.8635191143618253,-0.004910454235446947,0.9714000771679133,0.7124079513964765,0.08579660630312494,0.0,-0.11830288455326832,-0.06850886622136097,-0.23538907229979467,0.0,0.2799237988044963,0.250663775008682,-1.198513239776903,0.0,0.12362596854740969,-0.05878333264331127,-0.12692380550896032,-0.20403604380651194,0.6920828087351294,1.0515833779351837,-0.0801054696408375,0.0,0.5201672239033026,0.8551791352124384,0.8679443811693326,-0.2901320200591761,-0.09047535444288683,-0.610554930062075,1.2726965370004928,0.0,-0.5679124294045508,-0.1282763890923838,0.9108892719295474,0.6700785143230438 };
float bias_2[8]= { 0.4551804380254007,0.7229253848358435,-0.2669865522218793,-0.06172150051765611,0.5513902396918064,1.2611051426371567,0.6518310330866665,-0.026778355500840144 };
float weight_3[48]= { -0.051853004023389195,0.30762754571503803,1.2766120373680874,-0.010975122782634992,-0.11256699826929042,-0.20449250820060702,0.46681456293640494,0.4656124442231735,0.9438647438717637,0.09958938979181498,0.33218706628256534,0.7523120173055686,-0.2079952527783484,-0.7277459106107961,-2.1658203925945867,0.003685513105241172,0.02338347674460254,-0.016244169762622204,0.0,0.0,0.0,0.0,0.0,0.0,0.18231174633238775,0.32725877486720867,0.9269509941606056,0.12031569705900992,0.19939438143064409,0.38756154544772947,-0.49607400375896865,0.033643672244830516,1.0259267834516337,0.01643938392699229,0.14516751253577906,1.181522931663822,0.015052788527109067,-0.3444784317635556,-1.3494295598445956,0.13785095548427187,0.3434121002816258,0.8971563938989363,0.5713825278898487,1.1327500214510842,2.6365855948874075,-0.030154761617544035,-0.04319470634014424,0.0036258570549052606 };
float bias_3[8]= { 0.6401183008887498,0.389618968176662,0.8120740044278264,-0.33853292007385577,0.10634133882734984,0.8631314291536991 };
half weight_1_half[32], bias_1_half[8],weight_2_half[64], bias_2_half[8],weight_3_half[64], bias_3_half[8];
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
//end of tensorox_define, call prepare_half_prec_weights in the main function to initialize these arrays, then use cudaMemcpyToSymbol to copy the weight_i_half to weight_i_half_d
//tensorox_define

using namespace std;

__global__ void invkin_kernel(float *xTarget_in, float *yTarget_in, float *angles, int size, float err_thresh)
{

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

		 int tid = blockDim.x * blockIdx.x + threadIdx.x;
		 int idx = tid;

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
		 __shared__ half weight_3_shared[128];
		 __shared__ half bias_3_shared[256];
		 wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag_col;
    		 wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    		 wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

		 //define or set constant WMMA_M = 32, WMMA_N = 8, WMMA_K = 16 to use this mma_sync variant
		 weight_1_shared[real_tid] = 0.0;
    		 weight_2_shared[real_tid] = 0.0;
    		 weight_3_shared[real_tid] = 0.0;
    		 // 8 is the number of warps, simply reset buffer mem for all warps
    		 for (int i = 0; i<8 ; i++){
    		      neuron_out[i][real_tid] = 0.0;
    		      neuron_out[i][real_tid+256] = 0.0;
    		    }
    		 __syncthreads();
		// if condition to avoid illegal memaccess.
		 // can further reduce number of mem_load by checking this condition separately for each layer e.g. in case layer 1 only needs to load 16 or 32 weights.
		  if (real_tid < 32)
			 weight_1_shared[real_tid] = weight_1_half_d[real_tid];
		 if (real_tid <64){
			 weight_2_shared[real_tid] = weight_2_half_d[real_tid];
			 weight_3_shared[real_tid] = weight_3_half_d[real_tid];
		 }
		 bias_1_shared[real_tid] = bias_1_half_d[warp_id];
		 bias_2_shared[real_tid] = bias_2_half_d[warp_id];
		 bias_3_shared[real_tid] = bias_3_half_d[warp_id];
		 neuron_out[warp_id][warp_lane+ 0*32] = ( xTarget_in[2*tid] - -4.785800E-04 )* 8.009000E-01 ;
		 neuron_out[warp_id][warp_lane+ 1*32] = ( yTarget_in[2*tid] - 1.444300E+00 )* 1.493900E+00 ;
		 neuron_out[warp_id][warp_lane+ 2*32] = ( xTarget_in[2*tid+1] - -1.380800E-03 )* 8.002000E-01 ;
		 neuron_out[warp_id][warp_lane+ 3*32] = ( yTarget_in[2*tid+1] - 1.445000E+00 )* 1.495200E+00 ;
		 __syncthreads();
		 wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
		 wmma::load_matrix_sync(b_frag, (const __half*)&weight_1_shared, 8);
		 wmma::load_matrix_sync(c_frag, (const half*)&bias_1_shared, 32, wmma::mem_col_major);
		 wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
		 for (int i = 0; i< c_frag.num_elements; i ++)
			 c_frag.x[i] = relu(c_frag.x[i]);
		 wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);
		 // do not need __syncthreads() here because all instructions above are sync. End of layer 1
		 wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
		 wmma::load_matrix_sync(b_frag, (const __half*)&weight_2_shared, 8);
		 wmma::load_matrix_sync(c_frag, (const half*)&bias_2_shared, 32, wmma::mem_col_major);
		 wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
		 for (int i = 0; i< c_frag.num_elements; i ++)
			 c_frag.x[i] = relu(c_frag.x[i]);
		 wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);
		 // do not need __syncthreads() here because all instructions above are sync. End of layer 2
		 wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);
		 wmma::load_matrix_sync(b_frag, (const __half*)&weight_3_shared, 8);
		 wmma::load_matrix_sync(c_frag, (const half*)&bias_3_shared, 32, wmma::mem_col_major);
		 wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);
		 //performance trick: if the last layer only has 2-3 outputs, it maybe faster to compute dot product explicitly (e.g. 2 outputs cost 16 multiplying-accumulate ops) without using mma_sync
		 wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);
		 // do not need __syncthreads() here because all instructions above are sync. End of layer 3
		 angles[(2*tid)*3+0] = __half2float(neuron_out[warp_id][warp_lane+0*32])*1.000000E+02;
		 angles[(2*tid)*3+1] = __half2float(neuron_out[warp_id][warp_lane+1*32])*1.000000E+02;
		 angles[(2*tid)*3+2] = __half2float(neuron_out[warp_id][warp_lane+2*32])*1.000000E+02;
		 angles[(2*tid+1)*3+0] = __half2float(neuron_out[warp_id][warp_lane+3*32])*1.000000E+02;
		 angles[(2*tid+1)*3+1] = __half2float(neuron_out[warp_id][warp_lane+4*32])*1.000000E+02;
		 angles[(2*tid+1)*3+2] = __half2float(neuron_out[warp_id][warp_lane+5*32])*1.000000E+02;
		// end of Tensor approx region
}
int main(int argc, char* argv[])
{
	if(argc != 4)
	{
		std::cerr << "Usage: ./invkin.out <input file coefficients> <output file> <error threshold>" << std::endl;
		exit(EXIT_FAILURE);
	}

	float* xTarget_in_h;
	float* yTarget_in_h;
	float* angle_out_h;

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

	std::cout << "# Data are transfered to GPU..." << std::endl;

  dim3 dimBlock	( 256, 1 );
	dim3 dimGrid	( data_size / (256*2), 1 ); //256x5 for each block
//	dim3 dimGrid	( 2, 1 ); //256x5 for each block


	cudaEventRecord(start, 0);

#pragma parrot.start("invkin_kernel")

	invkin_kernel<<<dimGrid, dimBlock>>>(xTarget_in_d, yTarget_in_d, angle_out_d, data_size, err_thresh);

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

	{
		FILE *fp;
		fp = fopen("input_invk2j.txt", "w");
		for (int i = 0; i < data_size; i++)
		{
			 if (i == data_size -1 )
					fprintf(fp, "%f,%f",xTarget_in_h[i], yTarget_in_h[i]);
			 else
					fprintf(fp, "%f,%f,",xTarget_in_h[i], yTarget_in_h[i]);
		}
		fclose(fp);

	}

	{
		FILE *fp;
		fp = fopen("output_invk2j.txt", "w");
		for (int i = 0; i < data_size; i++)
		{
			 if (i == data_size -1 )
					fprintf(fp, "%f,%f,%f",angle_out_h[i*3],angle_out_h[i*3+1],angle_out_h[i*3+2] );
			 else
					fprintf(fp, "%f,%f,%f,",angle_out_h[i*3],angle_out_h[i*3+1],angle_out_h[i*3+2]);
		}
		fclose(fp);

	}

	for(int i = 0; i < data_size; i++)
	{
		angle_out_file << xTarget_in_h[i] << " " << yTarget_in_h[i] << " ";
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
