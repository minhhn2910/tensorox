#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef unsigned int uint;

#define checkCudaErrors(call)                                                  \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", __FILE__,        \
              __LINE__, #call, cudaGetErrorString(err__));                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

inline cudaError_t bindNormalizedUchar4Tex(cudaTextureObject_t *tex,
                                           const void *devPtr,
                                           const cudaChannelFormatDesc &desc,
                                           size_t width, size_t height,
                                           size_t pitch) {
  if (tex == 0) {
    return cudaErrorInvalidValue;
  }
  if (*tex) {
    cudaDestroyTextureObject(*tex);
    *tex = 0;
  }

  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = const_cast<void *>(devPtr);
  resDesc.res.pitch2D.desc = desc;
  resDesc.res.pitch2D.width = width;
  resDesc.res.pitch2D.height = height;
  resDesc.res.pitch2D.pitchInBytes = pitch;

  cudaTextureDesc texDesc = {};
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 0;

  return cudaCreateTextureObject(tex, &resDesc, &texDesc, 0);
}

inline cudaError_t unbindTex(cudaTextureObject_t *tex) {
  if (tex == 0 || *tex == 0) {
    return cudaSuccess;
  }
  cudaError_t err = cudaDestroyTextureObject(*tex);
  *tex = 0;
  return err;
}
