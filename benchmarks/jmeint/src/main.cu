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

#define EPSILON 1e-12 // EPSILON represents the error buffer used to denote a hit

using namespace std;


__device__ bool newComputeIntervals(float vv0, float vv1, float vv2, float d0, float d1, float d2, float d0d1, float d0d2, float abc[3], float x0x1[2])
{
	if (d0d1 > 0.0f) {
		// d0d2 <= 0 --> i.e. d0, d1 are on the same side, d2 on the other or on the plane
		abc[0] = vv2;
		abc[1] = (vv0 - vv2) * d2;
		abc[2] = (vv1 - vv2) * d2;
		x0x1[0] = d2 - d0;
		x0x1[1] = d2 - d1;
	} else if (d0d2 > 0.0f) {
		// d0d1 <= 0
		abc[0] = vv1;
		abc[1] = (vv0 - vv1) * d1;
		abc[2] = (vv2 - vv1) * d1;
		x0x1[0] = d1 - d0;
		x0x1[1] = d1 - d2;
	} else if (d1 * d2 > 0.0f || d0 != 0.0f) {
		// d0d1 <= 0 or d0 != 0
		abc[0] = vv0;
		abc[1] = (vv1 - vv0) * d0;
		abc[2] = (vv2 - vv0) * d0;
		x0x1[0] = d0 - d1;
		x0x1[1] = d0 - d2;
	} else if (d1 != 0.0f) {
		abc[0] = vv1;
		abc[1] = (vv0 - vv1) * d1;
		abc[2] = (vv2 - vv1) * d1;
		x0x1[0] = d1 - d0;
		x0x1[1] = d1 - d2;
	} else if (d2 != 0.0f) {
		abc[0] = vv2;
		abc[1] = (vv0 - vv2) * d2;
		abc[2] = (vv1 - vv2) * d2;
		x0x1[0] = d2 - d0;
		x0x1[1] = d2 - d1;
	} else {
		// Triangles are coplanar
		return true;
	}

	return false;
}


__device__ bool edgeEdgeTest(float v0[3], float u0[3], float u1[3], int i0, int i1, float Ax, float Ay)
{
	float Bx = u0[i0] - u1[i0];
	float By = u0[i1] - u1[i1];
	float Cx = v0[i0] - u0[i0];
	float Cy = v0[i1] - u0[i1];
	float f = Ay * Bx - Ax * By;
	float d = By * Cx - Bx * Cy;

	if ((f > 0 && d >= 0 && d <= f) || (f < 0 && d <= 0 && d >= f)) {
		float e = Ax * Cy - Ay * Cx;
		if (f > 0) {
			if (e >= 0 && e <= f)
				return true;
		} else {
			if (e <= 0 && e >= f)
				return true;
		}
	}

	return false;
}


__device__ bool pointInTri(float V0[3], float U0[3], float U1[3], float U2[3], int i0, int i1)
{
	// Check if V0 is inside triangle (U0,U1,U2)

	float a, b, c, d0, d1, d2;
	a = U1[i1] - U0[i1];
	b = -(U1[i0] - U0[i0]);
	c = -a * U0[i0] - b * U0[i1];
	d0 = a * V0[i0] + b * V0[i1] + c;

	a = U2[i1] - U1[i1];
	b = -(U2[i0] - U1[i0]);
	c = -a * U1[i0] - b * U1[i1];
	d1 = a * V0[i0] + b * V0[i1] + c;

	a = U0[i1] - U2[i1];
	b = -(U0[i0] - U2[i0]);
	c = -a * U2[i0] - b * U2[i1];
	d2 = a * V0[i0] + b * V0[i1] + c;

	if ((d0 * d1) > 0.0 && (d0 * d2) > 0.0)
		return true;

	return false;
}


__device__ bool coplanarTriTri(float n[3], float v0[3], float v1[3], float v2[3], float u0[3], float u1[3], float u2[3])
{
	float a[3];
	short i0, i1;
	a[0] = abs(n[0]);
	a[1] = abs(n[1]);
	a[2] = abs(n[2]);

	if (a[0] > a[1]) {
		if (a[0] > a[2]) {
			i0 = 1;
			i1 = 2;
		} else {
			i0 = 0;
			i1 = 1;
		}
	} else {
		if (a[2] > a[1]) {
			i0 = 0;
			i1 = 1;
		} else {
			i0 = 0;
			i1 = 2;
		}
	}

	// Test all edges of triangle 1 against edges of triangle 2
	float aX = v1[i0] - v0[i0];
	float aY = v1[i1] - v0[i1];
	float bX = v2[i0] - v1[i0];
	float bY = v2[i1] - v1[i1];
	float cX = v0[i0] - v2[i0];
	float cY = v0[i1] - v2[i1];
	if ( edgeEdgeTest(v0, u0, u1, i0, i1, aX, aY) || edgeEdgeTest(v0, u1, u2, i0, i1, aX, aY) || edgeEdgeTest(v0, u2, u0, i0, i1, aX, aY) ||
		 edgeEdgeTest(v1, u0, u1, i0, i1, bX, bY) || edgeEdgeTest(v1, u1, u2, i0, i1, bX, bY) || edgeEdgeTest(v1, u2, u0, i0, i1, bX, bY) ||
		 edgeEdgeTest(v2, u0, u1, i0, i1, cX, cY) || edgeEdgeTest(v2, u1, u2, i0, i1, cX, cY) || edgeEdgeTest(v2, u2, u0, i0, i1, cX, cY) )
		return true;

	// Finally, test if either triangle is totally contained in the other
	if (pointInTri(v0, u0, u1, u2, i0, i1) || pointInTri(u0, v0, v1, v2, i0, i1))
		return true;
	return false;

}

__device__ bool jmeint_kernel_impl(float v0[3], float v1[3], float v2[3], float u0[3], float u1[3], float u2[3])
{

    	float e1[3], e2[3], n1[3], n2[3], d[3];
    	float d1, d2;
    	float du0, du1, du2, dv0, dv1, dv2;
    	float du0du1, du0du2, dv0dv1, dv0dv2;

    	float isect1[2];
		float isect2[2];
		short index;
		float vp0, vp1, vp2;
		float up0, up1, up2;
		float bb, cc, max;
		float xx, yy, xxyy, tmp;

		// Compute plane equation of triangle (v0,v1,v2)
		e1[0] = v1[0] - v0[0];
		e1[1] = v1[1] - v0[1];
		e1[2] = v1[2] - v0[2];

		e2[0] = v2[0] - v0[0];
		e2[1] = v2[1] - v0[1];
		e2[2] = v2[2] - v0[2];

		// Cross product: n1 = e1 x e2
		n1[0] = (e1[1] * e2[2]) - (e1[2] * e2[1]);
		n1[1] = (e1[2] * e2[0]) - (e1[0] * e2[2]);
		n1[2] = (e1[0] * e2[1]) - (e1[1] * e2[0]);

		// Plane equation 1: n1.X + d1 = 0
		d1 = -(n1[0] * v0[0] + n1[1] * v0[1] + n1[2] * v0[2]);

		// Put u0,u1,u2 into plane equation 1 to compute signed distances to the plane
		du0 = (n1[0] * u0[0] + n1[1] * u0[1] + n1[2] * u0[2]) + d1;
		du1 = (n1[0] * u1[0] + n1[1] * u1[1] + n1[2] * u1[2]) + d1;
		du2 = (n1[0] * u2[0] + n1[1] * u2[1] + n1[2] * u2[2]) + d1;

		// Coplanarity robustness check
		if ((du0 > 0 && du0 < EPSILON) || (du0 < 0 && du0 > EPSILON))
			du0 = 0.0f;
		if ((du1 > 0 && du1 < EPSILON) || (du1 < 0 && du1 > EPSILON))
			du1 = 0.0f;
		if ((du2 > 0 && du2 < EPSILON) || (du2 < 0 && du2 > EPSILON))
			du2 = 0.0f;

		du0du1 = du0 * du1;
		du0du2 = du0 * du2;

		if (du0du1 > 0.0f && du0du2 > 0.0f) {
			// All 3 have same sign and their values are not equal to 0 --> no intersection
			return false;
		}

		// Compute plane equation of triangle (u0,u1,u2)
		e1[0] = u1[0] - u0[0];
		e1[1] = u1[1] - u0[1];
		e1[2] = u1[2] - u0[2];

		e2[0] = u2[0] - u0[0];
		e2[1] = u2[1] - u0[1];
		e2[2] = u2[2] - u0[2];

		// Cross product: n2 = e1 x e2
		n2[0] = (e1[1] * e2[2]) - (e1[2] * e2[1]);
		n2[1] = (e1[2] * e2[0]) - (e1[0] * e2[2]);
		n2[2] = (e1[0] * e2[1]) - (e1[1] * e2[0]);

		// Plane equation 2: n2.X + d2 = 0
		d2 = -(n2[0] * u0[0] + n2[1] * u0[1] + n2[2] * u0[2]);

		// Put v0,v1,v2 into plane equation 2 to compute signed distances to the plane
		dv0 = (n2[0] * v0[0] + n2[1] * v0[1] + n2[2] * v0[2]) + d2;
		dv1 = (n2[0] * v1[0] + n2[1] * v1[1] + n2[2] * v1[2]) + d2;
		dv2 = (n2[0] * v2[0] + n2[1] * v2[1] + n2[2] * v2[2]) + d2;

		// Coplanarity robustness check
		if ((dv0 > 0 && dv0 < EPSILON) || (dv0 < 0 && dv0 > EPSILON))
			dv0 = 0.0f;
		if ((dv1 > 0 && dv1 < EPSILON) || (dv1 < 0 && dv1 > EPSILON))
			dv1 = 0.0f;
		if ((dv2 > 0 && dv2 < EPSILON) || (dv2 < 0 && dv2 > EPSILON))
			dv2 = 0.0f;

		dv0dv1 = dv0 * dv1;
		dv0dv2 = dv0 * dv2;

		if (dv0dv1 > 0.0f && dv0dv2 > 0.0f) {
			// All 3 have same sign and their values are not equal to 0 --> no intersection
			return false;
		}
		// Compute direction of intersection line --> cross product: d = n1 x n2
		d[0] = (n1[1] * n2[2]) - (n1[2] * n2[1]);
    	d[1] = (n1[2] * n2[0]) - (n1[0] * n2[2]);
    	d[2] = (n1[0] * n2[1]) - (n1[1] * n2[0]);

		// Compute and index to the largest component of d
		index = 0;
		max = abs(d[0]);
		bb = abs(d[1]);
		cc = abs(d[2]);
		if (bb > max) {
			max = bb;
			index = 1;
		}
		if (cc > max) {
			max = cc;
			vp0 = v0[2];
			vp1 = v1[2];
			vp2 = v2[2];
			up0 = u0[2];
			up1 = u1[2];
			up2 = u2[2];
		} else if (index == 1) {
			vp0 = v0[1];
			vp1 = v1[1];
			vp2 = v2[1];
			up0 = u0[1];
			up1 = u1[1];
			up2 = u2[1];
		} else {
			vp0 = v0[0];
			vp1 = v1[0];
			vp2 = v2[0];
			up0 = u0[0];
			up1 = u1[0];
			up2 = u2[0];
		}

		// Compute interval for triangle 1
		float abc[3];
		float x0x1[2];
		if (newComputeIntervals(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, abc, x0x1)) {
			return coplanarTriTri(n1, v0, v1, v2, u0, u1, u2);
		}

		// Compute interval for triangle 2
		float def[3];
		float y0y1[2];
		if (newComputeIntervals(up0, up1, up2, du0, du1, du2, du0du1, du0du2, def, y0y1)) {
			return coplanarTriTri(n1, v0, v1, v2, u0, u1, u2);
		}
		xx = x0x1[0] * x0x1[1];
		yy = y0y1[0] * y0y1[1];
		xxyy = xx * yy;

		tmp = abc[0] * xxyy;
		isect1[0] = tmp + abc[1] * x0x1[1] * yy;
		isect1[1] = tmp + abc[2] * x0x1[0] * yy;

		tmp = def[0] * xxyy;
		isect2[0] = tmp + def[1] * xx * y0y1[1];
		isect2[1] = tmp + def[2] * xx * y0y1[0];

		// Sort isect1 and isect2
		if (isect1[0] > isect1[1]) {
			float f = isect1[0];
			isect1[0] = isect1[1];
			isect1[1] = f;
		}
		if (isect2[0] > isect2[1]) {
			float f = isect2[0];
			isect2[0] = isect2[1];
			isect2[1] = f;
		}

		if (isect1[1] < isect2[0] || isect2[1] < isect1[0])
		{
			return false;
		}
		return true;
}


__global__ void jmeint_kernel(float *v0_d, float *v1_d, float *v2_d, float *u0_d, float*u1_d, float*u2_d, bool* intersect_d, int size)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


	float v0[3];
	float v1[3];
	float v2[3];
	float u0[3];
	float u1[3];
	float u2[3];

	if(idx < size)
	{

		v0[0] = v0_d[idx * 3 + 0];
		v0[1] =	v0_d[idx * 3 + 1];
		v0[2] = v0_d[idx * 3 + 2];

		v1[0] =	v1_d[idx * 3 + 0];
		v1[1] =	v1_d[idx * 3 + 1];
		v1[2] =	v1_d[idx * 3 + 2];

		v2[0] = v2_d[idx * 3 + 0];
		v2[1] = v2_d[idx * 3 + 1];
		v2[2] =	v2_d[idx * 3 + 2];

		u0[0] = u0_d[idx * 3 + 0];
		u0[1] =	u0_d[idx * 3 + 1];
		u0[2] = u0_d[idx * 3 + 2];

		u1[0] = u1_d[idx * 3 + 0];
		u1[1] = u1_d[idx * 3 + 1];
		u1[2] = u1_d[idx * 3 + 2];

		u2[0] = u2_d[idx * 3 + 0];
		u2[1] = u2_d[idx * 3 + 1];
		u2[2] = u2_d[idx * 3 + 2];

		float parrotInput[18];
    	float parrotOutput[1];

    	parrotInput[0 ] = v0[0];
    	parrotInput[1 ] = v0[1];
    	parrotInput[2 ] = v0[2];

    	parrotInput[3 ] = v1[0];
    	parrotInput[4 ] = v1[1];
    	parrotInput[5 ] = v1[2];

    	parrotInput[6 ] = v2[0];
    	parrotInput[7 ] = v2[1];
    	parrotInput[8 ] = v2[2];

    	parrotInput[9 ] = u0[0];
    	parrotInput[10] = u0[1];
    	parrotInput[11] = u0[2];

    	parrotInput[12] = u1[0];
    	parrotInput[13] = u1[1];
    	parrotInput[14] = u1[2];

    	parrotInput[15] = u2[0];
    	parrotInput[16] = u2[1];
    	parrotInput[17] = u2[2];

#pragma parrot(input, "jmeint_kernel", [18]<-1.0; 1.0>parrotInput)

   		intersect_d[idx] = jmeint_kernel_impl(v0, v1, v2, u0, u1, u2);

   		if(intersect_d[idx])
   		{
   			parrotOutput[0] = -0.9;
   		}
   		else
   		{
   			parrotOutput[0] = 0.9;
   		}

#pragma parrot(output, "jmeint_kernel", [1]<-0.9; 0.9>parrotOutput)


		if(parrotOutput[0] > 0.0)
		{
			intersect_d[idx] = true;
			return;
		}
		else
		{
			intersect_d[idx] = false;
			return;
		}

	}
}

int main(int argc, char* argv[])
{
	if(argc != 3)
	{
		std::cerr << "Usage: ./jmeint.out <input file locations> <output file>" << std::endl;
		exit(EXIT_FAILURE);
	}

	float (*v0)[3];
	float (*v1)[3];
	float (*v2)[3];
	float (*u0)[3];
	float (*u1)[3];
	float (*u2)[3];
	bool  *intersect;


	cudaError_t cudaStatus;

	int data_size = 0;

	// process the files
	ifstream locations_in_file (argv[1]);
	ofstream intersect_out_file (argv[2]);


	if(locations_in_file.is_open())
	{
		locations_in_file >> data_size;
		std::cout << "# Data Size = " << data_size << std::endl;
	}


	intersect = new (nothrow) bool[data_size];

	// allocate the memory
	v0 = new (nothrow) float[data_size][3];
	if(v0 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	v1 = new (nothrow) float[data_size][3];
	if(v1 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	v2 = new (nothrow) float[data_size][3];
	if(v2 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	u0 = new (nothrow) float[data_size][3];
	if(u0 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	u1 = new (nothrow) float[data_size][3];
	if(u1 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}
	// allocate the memory
	u2 = new (nothrow) float[data_size][3];
	if(u2 == NULL)
	{
		std::cerr << "Memory allocation fails!!!" << std::endl;
		exit(EXIT_FAILURE);
	}



	// Prepare
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// add data to the arrays
	int loc_index = 0;
	while(loc_index < data_size)
	{
		locations_in_file 	>> v0[loc_index][0] >> v0[loc_index][1] >> v0[loc_index][2]
							>> v1[loc_index][0] >> v1[loc_index][1] >> v1[loc_index][2]
							>> v2[loc_index][0] >> v2[loc_index][1] >> v2[loc_index][2]
							>> u0[loc_index][0] >> u0[loc_index][1] >> u0[loc_index][2]
							>> u1[loc_index][0] >> u1[loc_index][1] >> u1[loc_index][2]
							>> u2[loc_index][0] >> u2[loc_index][1] >> u2[loc_index][2];
		loc_index++;
	}


	std::cout << "# Coordinates are read from file..." << std::endl;

	// memory allocations on the host
	float *v0_d;
	float *v1_d;
	float *v2_d;
	float *u0_d;
	float *u1_d;
	float *u2_d;
	bool  *intersect_d;

	cudaMalloc((void**) &v0_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &v1_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &v2_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &u0_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &u1_d, data_size * 3 * sizeof(float));
	cudaMalloc((void**) &u2_d, data_size * 3 * sizeof(float));

	cudaMalloc((void**) &intersect_d, data_size * sizeof(bool));


	std::cout << "# Memory allocation on GPU is done..." << std::endl;

	cudaMemcpy(v0_d, v0, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(v1_d, v1, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(v2_d, v2, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(u0_d, u0, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(u1_d, u1, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(u2_d, u2, data_size * 3 * sizeof(float), cudaMemcpyHostToDevice);


	std::cout << "# Data are transfered to GPU..." << std::endl;

	dim3 dimBlock	( 256, 1 );
	dim3 dimGrid	( data_size / 256, 1 );


	cudaEventRecord(start, 0);

#pragma parrot.start("jmeint_kernel")

	jmeint_kernel<<<dimGrid, dimBlock>>>(v0_d, v1_d, v2_d, u0_d, u1_d, u2_d, intersect_d, data_size);

#pragma parrot.end("jmeint_kernel")

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

	std::cout << "# Elapsed Time in `jmeint` kernel = " << elapsedTime << std::endl;
	std::cout << "# GPU computation is done ..." << std::endl;

	cudaMemcpy(intersect, intersect_d, data_size * sizeof(bool), cudaMemcpyDeviceToHost);

	for(int i = 0; i < data_size; i++)
	{
		intersect_out_file << intersect[i];
		intersect_out_file << std::endl;
	}

	// close files
	locations_in_file.close();
	intersect_out_file.close();

	// de-allocate the memory
	delete[] v0;
	delete[] v1;
	delete[] v2;
	delete[] u0;
	delete[] u1;
	delete[] u2;
	delete[] intersect;

	// de-allocate cuda memory
	cudaFree(v0_d);
	cudaFree(v1_d);
	cudaFree(v2_d);
	cudaFree(u0_d);
	cudaFree(u1_d);
	cudaFree(u2_d);
	cudaFree(intersect_d);

	std::cout << "Thank you..." << std::endl;
}
