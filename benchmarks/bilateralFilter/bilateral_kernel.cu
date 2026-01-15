#include "cuda_utils.h"
#include <math.h>
#include <vector_types.h>

__constant__ float cGaussian[64];
static cudaTextureObject_t rgbaTex = 0;

uint *dImage = NULL;
uint *dTemp = NULL;
size_t pitch;

__device__ float euclideanLen(float4 a, float4 b, float d) {
  float mod = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) +
              (b.z - a.z) * (b.z - a.z);
  return __expf(-mod / (2.f * d * d));
}

__device__ uint rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(fabsf(rgba.x));
  rgba.y = __saturatef(fabsf(rgba.y));
  rgba.z = __saturatef(fabsf(rgba.z));
  rgba.w = __saturatef(fabsf(rgba.w));
  return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) |
         (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__global__ void d_bilateral_filter(uint *od, int w, int h, float e_d, int r,
                                   cudaTextureObject_t tex) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) {
    return;
  }

  float sum = 0.0f;
  float4 t = {0.f, 0.f, 0.f, 0.f};
  float4 center = tex2D<float4>(tex, x, y);

  for (int i = -r; i <= r; i++) {
    for (int j = -r; j <= r; j++) {
      float4 curPix = tex2D<float4>(tex, x + j, y + i);
      float factor = cGaussian[i + r] * cGaussian[j + r] *
                     euclideanLen(curPix, center, e_d);
      t.x += factor * curPix.x;
      t.y += factor * curPix.y;
      t.z += factor * curPix.z;
      t.w += factor * curPix.w;
      sum += factor;
    }
  }

  float inv = 1.0f / sum;
  t.x *= inv;
  t.y *= inv;
  t.z *= inv;
  t.w *= inv;
  od[y * w + x] = rgbaFloatToInt(t);
}

extern "C" void initTexture(int width, int height, uint *hImage) {
  checkCudaErrors(
      cudaMallocPitch(&dImage, &pitch, sizeof(uint) * width, height));
  checkCudaErrors(
      cudaMallocPitch(&dTemp, &pitch, sizeof(uint) * width, height));
  checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(uint) * width,
                               sizeof(uint) * width, height,
                               cudaMemcpyHostToDevice));
}

extern "C" void freeTextures() {
  checkCudaErrors(unbindTex(&rgbaTex));
  checkCudaErrors(cudaFree(dImage));
  checkCudaErrors(cudaFree(dTemp));
}

extern "C" void updateGaussian(float delta, int radius) {
  float fGaussian[64];
  for (int i = 0; i < 2 * radius + 1; ++i) {
    float x = (float)(i - radius);
    fGaussian[i] = expf(-(x * x) / (2 * delta * delta));
  }
  checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian,
                                     sizeof(float) * (2 * radius + 1)));
}

extern "C" double bilateralFilterRGBA(uint *dDest, int width, int height,
                                      float e_d, int radius, int iterations) {
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
  checkCudaErrors(
      bindNormalizedUchar4Tex(&rgbaTex, dImage, desc, width, height, pitch));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  float totalMs = 0.f;
  for (int i = 0; i < iterations; i++) {
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start));
    dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
    dim3 blockSize(16, 16);
    d_bilateral_filter<<<gridSize, blockSize>>>(dDest, width, height, e_d,
                                                radius, rgbaTex);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    float ms = 0.f;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
    totalMs += ms;

    if (iterations > 1) {
      checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int) * width,
                                   sizeof(int) * width, height,
                                   cudaMemcpyDeviceToDevice));
      checkCudaErrors(bindNormalizedUchar4Tex(&rgbaTex, dTemp, desc, width,
                                              height, pitch));
    }
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  return (totalMs / 1000.0) / (double)iterations;
}
