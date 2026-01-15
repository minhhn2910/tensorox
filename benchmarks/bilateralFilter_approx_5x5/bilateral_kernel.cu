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

#include "cuda_utils.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <vector_functions.h>
#include <math.h>
#include <cuda.h>
using namespace nvcuda;
// The only dimensions currently supported by WMMA
const int WMMA_M = 32;
const int WMMA_N = 8;
const int WMMA_K = 16;
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define SKEW_HALF 8 // pad WMMA leading dim so LDSM is 1-way
#define LDA (WMMA_K + SKEW_HALF)
#define TILE_W 20
#define TILE_H 20
#define TILE_PIX (TILE_W * TILE_H)

__constant__ __align__(16) half weight_1_half_d[256];
__constant__ __align__(16) half weight_2_half_d[24];

__constant__ __align__(16) half bias_1_half_d[8];
__constant__ __align__(16) half bias_2_half_d[8];

__constant__ float means_d[28];
__constant__ float scale_d[28];

__constant__ float cGaussian[64];   //gaussian array in device side
static cudaTextureObject_t rgbaTex = 0;

uint *dImage  = NULL;   //original image
uint *dTemp   = NULL;   //temp array for iterations
uint *dPadded = NULL;   // clamp-padded image for Hopper TMA
size_t pitch;
size_t padded_pitch = 0;
int padded_w = 0;
int padded_h = 0;
static CUtensorMap g_tmap;
static int g_sm_major = 0;

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
__device__ __forceinline__ half relu(half x){
  return __hmax(x, __float2half_rn(0.f));
}

__device__ __forceinline__ unsigned pack_half2(__half2 v)
{
    return *reinterpret_cast<const unsigned *>(&v);
}

__device__ __forceinline__ uint4 splat_half8(half v)
{
    unsigned p = pack_half2(__half2half2(v));
    return make_uint4(p, p, p, p);
}

__device__ __forceinline__ void st16(half *row, uint4 a, uint4 b)
{
    uint4 *p = reinterpret_cast<uint4 *>(row);
    p[0] = a;
    p[1] = b;
}

__device__ __forceinline__ void pack_rgb(half *f, int k, float4 pix, int mi)
{
    f[k]     = __float2half_rn((pix.x - means_d[mi])     * scale_d[mi]);
    f[k + 1] = __float2half_rn((pix.y - means_d[mi + 1]) * scale_d[mi + 1]);
    f[k + 2] = __float2half_rn((pix.z - means_d[mi + 2]) * scale_d[mi + 2]);
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
                   float e_d,  int r, cudaTextureObject_t tex)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;


    //debug
  //  if (blockIdx.x >1 || blockIdx.y  >1 )  return ;
  //  if (blockIdx.x >3 || blockIdx.y > 3)
  // return;
  //  int index = y*w +x;
  //  int index = x*h+y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int real_tid = ty * blockDim.x + tx;
    const int warp_id = real_tid / 32;
    const int warp_lane = real_tid % 32;
    const int base_x = blockIdx.x * blockDim.x;
    const int base_y = blockIdx.y * blockDim.y;

    // One WMMA A staging buffer per warp (row-major, LDA padded). K-tiles are
    // packed sequentially so we never zero the whole array.
    __shared__ __align__(16) half A[WARPS_PER_BLOCK][WMMA_M * LDA];
    __shared__ __align__(16) float4 sh_tile[TILE_PIX];
    __shared__ __align__(16) half weight_1_shared[2][128];
    __shared__ __align__(16) half bias_1_shared[256];
    __shared__ __align__(16) half weight_2_shared[128];
    __shared__ __align__(16) half bias_2_shared[256];

    if (real_tid < 32) {
      reinterpret_cast<uint4 *>(weight_1_shared)[real_tid] =
          reinterpret_cast<const uint4 *>(weight_1_half_d)[real_tid];
      reinterpret_cast<uint4 *>(bias_1_shared)[real_tid] =
          splat_half8(bias_1_half_d[real_tid / 4]);
      reinterpret_cast<uint4 *>(bias_2_shared)[real_tid] =
          splat_half8(bias_2_half_d[real_tid / 4]);
    }
    if (real_tid < 16) {
      uint4 w2 = make_uint4(0, 0, 0, 0);
      if (real_tid < 8) {
        __half2 a = __halves2half2(weight_2_half_d[real_tid * 3 + 0],
                                   weight_2_half_d[real_tid * 3 + 1]);
        __half2 b = __halves2half2(weight_2_half_d[real_tid * 3 + 2],
                                   __float2half_rn(0.f));
        w2 = make_uint4(pack_half2(a), pack_half2(b), 0, 0);
      }
      reinterpret_cast<uint4 *>(weight_2_shared)[real_tid] = w2;
    }

    {
      const int lx = real_tid % TILE_W;
      const int ly = real_tid / TILE_W;
      sh_tile[real_tid] =
          tex2D<float4>(tex, base_x + lx - 2, base_y + ly - 2);
    }
    if (real_tid < TILE_PIX - THREADS_PER_BLOCK) {
      const int idx = real_tid + THREADS_PER_BLOCK;
      const int lx = idx % TILE_W;
      const int ly = idx / TILE_W;
      sh_tile[idx] = tex2D<float4>(tex, base_x + lx - 2, base_y + ly - 2);
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    half *a_row = &A[warp_id][warp_lane * LDA];

    // K-tile 0: samples 0-4 (15 features + 1 pad zero inside the uint4 store)
    {
      uint4 pk[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
      half *f = reinterpret_cast<half *>(pk);
      pack_rgb(f, 0,  sh_tile[(ty + 0) * TILE_W + (tx + 0)], 0);
      pack_rgb(f, 3,  sh_tile[(ty + 0) * TILE_W + (tx + 2)], 3);
      pack_rgb(f, 6,  sh_tile[(ty + 0) * TILE_W + (tx + 4)], 6);
      pack_rgb(f, 9,  sh_tile[(ty + 2) * TILE_W + (tx + 0)], 9);
      pack_rgb(f, 12, sh_tile[(ty + 2) * TILE_W + (tx + 2)], 12);
      st16(a_row, pk[0], pk[1]);
    }
    __syncwarp();

    wmma::load_matrix_sync(a_frag, (const __half *)A[warp_id], LDA);
    wmma::load_matrix_sync(b_frag, (const __half *)&weight_1_shared[0], 8);
    wmma::load_matrix_sync(c_frag, (const half *)&bias_1_shared, 32, wmma::mem_col_major);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // K-tile 1: samples 5-8 (12 features + 4 pad zeros)
    {
      uint4 pk[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
      half *f = reinterpret_cast<half *>(pk);
      pack_rgb(f, 0, sh_tile[(ty + 2) * TILE_W + (tx + 4)], 16);
      pack_rgb(f, 3, sh_tile[(ty + 4) * TILE_W + (tx + 0)], 19);
      pack_rgb(f, 6, sh_tile[(ty + 4) * TILE_W + (tx + 2)], 22);
      pack_rgb(f, 9, sh_tile[(ty + 4) * TILE_W + (tx + 4)], 25);
      st16(a_row, pk[0], pk[1]);
    }
    __syncwarp();

    wmma::load_matrix_sync(a_frag, (const __half *)A[warp_id], LDA);
    wmma::load_matrix_sync(b_frag, (const __half *)&weight_1_shared[1], 8);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    for (int i = 0; i < c_frag.num_elements; i++)
      c_frag.x[i] = relu(c_frag.x[i]);
    wmma::store_matrix_sync((half *)A[warp_id], c_frag, LDA, wmma::mem_row_major);

    wmma::load_matrix_sync(a_frag, (const __half *)A[warp_id], LDA);
    wmma::load_matrix_sync(b_frag, (const __half *)&weight_2_shared, 8);
    wmma::load_matrix_sync(c_frag, (const half *)&bias_2_shared, 32, wmma::mem_col_major);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync((half *)A[warp_id], c_frag, 8, wmma::mem_row_major);

    __syncwarp();
    float4 output;
    output.x = A[warp_id][warp_lane * 8 + 0];
    output.y = A[warp_id][warp_lane * 8 + 1];
    output.z = A[warp_id][warp_lane * 8 + 2];
    if (x >= w || y >= h)
    {
        return;
    }

    od[y * w + x] = rgbaFloatToInt(output);
}

// ---------------------------------------------------------------------------
// Hopper warp specialization (SM 90+):
//   9-warp CTA (288 threads). Warp 8 is the TMA producer (double-buffered
//   20x20 halo). Warps 0-7 consume a 16x16 pixel tile and run the WMMA MLP.
//   Persistent grid-stride over tiles, aimed at 4K/8K frames.
// ---------------------------------------------------------------------------
#define PRODUCER_WARP 8
#define HOPPER_THREADS (WARP_SIZE * (WARPS_PER_BLOCK + 1))
#define TMA_STAGES 2
#define TILE_BYTES (TILE_PIX * (int)sizeof(uint))
// Each TMA dest must be 128B-aligned. 20*20*4=1600 is 64B off a 128B boundary.
#define TILE_STRIDE_U32 (TILE_PIX + 16)

struct __align__(128) TmaTile {
    uint pix[TILE_PIX];
    uint pad[16];
};
static_assert(sizeof(TmaTile) % 128 == 0, "TMA dest stages must be 128B-strided");

__global__ void pad_clamp_kernel(const uint *src, int w, int h, size_t src_pitch,
                                 uint *dst, int pw, int ph, size_t dst_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pw || y >= ph) {
        return;
    }
    int sx = x - 2;
    int sy = y - 2;
    if (sx < 0) sx = 0;
    if (sy < 0) sy = 0;
    if (sx >= w) sx = w - 1;
    if (sy >= h) sy = h - 1;
    const uint *srow = (const uint *)((const char *)src + sy * src_pitch);
    uint *drow = (uint *)((char *)dst + y * dst_pitch);
    drow[x] = srow[sx];
}

#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ unsigned cvta_smem(const void *p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t *mbar, unsigned count)
{
    unsigned a = cvta_smem(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count)
                 : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t *mbar, unsigned bytes)
{
    unsigned a = cvta_smem(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;" ::"r"(
            a),
        "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t *mbar)
{
    unsigned a = cvta_smem(mbar);
    asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" ::"r"(a)
                 : "memory");
}

__device__ __forceinline__ void mbar_wait_parity(uint64_t *mbar, unsigned parity)
{
    unsigned a = cvta_smem(mbar);
    unsigned done = 0;
    while (!done) {
        asm volatile("{\n\t"
                     ".reg .pred p;\n\t"
                     "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 "
                     "p, [%1], %2;\n\t"
                     "selp.u32 %0, 1, 0, p;\n\t"
                     "}\n"
                     : "=r"(done)
                     : "r"(a), "r"(parity)
                     : "memory");
    }
}

__device__ __forceinline__ void tma_load_2d(void *dst, const CUtensorMap *tmap,
                                            int x, int y, uint64_t *mbar)
{
    unsigned ds = cvta_smem(dst);
    unsigned mb = cvta_smem(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::"
        "bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(ds),
        "l"(tmap), "r"(x), "r"(y), "r"(mb)
        : "memory");
}

__device__ __forceinline__ float4 u32_to_rgba(uint p)
{
    const float s = 0.003921568627f;
    return make_float4((p & 0xff) * s, ((p >> 8) & 0xff) * s,
                       ((p >> 16) & 0xff) * s, ((p >> 24) & 0xff) * s);
}
#endif

__global__ void __launch_bounds__(HOPPER_THREADS)
d_bilateral_filter_hopper(uint *od, int w, int h,
                          const __grid_constant__ CUtensorMap tmap)
{
#if __CUDA_ARCH__ >= 900
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_lane = tid & 31;
    const int is_producer = (warp_id == PRODUCER_WARP);

    const int tiles_x = (w + 15) >> 4;
    const int tiles_y = (h + 15) >> 4;
    const int ntiles = tiles_x * tiles_y;
    const int stride = gridDim.x;

    __shared__ __align__(16) half A[WARPS_PER_BLOCK][WMMA_M * LDA];
    __shared__ TmaTile smem_tile[TMA_STAGES];
    __shared__ __align__(8) uint64_t full_bar[TMA_STAGES];
    __shared__ __align__(8) uint64_t empty_bar[TMA_STAGES];
    __shared__ __align__(16) half weight_1_shared[2][128];
    __shared__ __align__(16) half bias_1_shared[256];
    __shared__ __align__(16) half weight_2_shared[128];
    __shared__ __align__(16) half bias_2_shared[256];

    if (tid < 32) {
        reinterpret_cast<uint4 *>(weight_1_shared)[tid] =
            reinterpret_cast<const uint4 *>(weight_1_half_d)[tid];
        reinterpret_cast<uint4 *>(bias_1_shared)[tid] =
            splat_half8(bias_1_half_d[tid / 4]);
        reinterpret_cast<uint4 *>(bias_2_shared)[tid] =
            splat_half8(bias_2_half_d[tid / 4]);
    }
    if (tid < 16) {
        uint4 w2 = make_uint4(0, 0, 0, 0);
        if (tid < 8) {
            __half2 a = __halves2half2(weight_2_half_d[tid * 3 + 0],
                                       weight_2_half_d[tid * 3 + 1]);
            __half2 b = __halves2half2(weight_2_half_d[tid * 3 + 2],
                                       __float2half_rn(0.f));
            w2 = make_uint4(pack_half2(a), pack_half2(b), 0, 0);
        }
        reinterpret_cast<uint4 *>(weight_2_shared)[tid] = w2;
    }
    if (tid == 0) {
        mbar_init(&full_bar[0], 1);
        mbar_init(&full_bar[1], 1);
        mbar_init(&empty_bar[0], WARPS_PER_BLOCK);
        mbar_init(&empty_bar[1], WARPS_PER_BLOCK);
    }
    __syncthreads();

    // Seed both buffers as empty so the producer can prefetch two tiles.
    if (!is_producer && warp_lane == 0) {
        mbar_arrive(&empty_bar[0]);
        mbar_arrive(&empty_bar[1]);
    }
    __syncthreads();

    if (is_producer) {
        unsigned empty_par[TMA_STAGES] = {0, 0};
        int i = 0;
        for (int t = (int)blockIdx.x; t < ntiles; t += stride, ++i) {
            const int stage = i & 1;
            // TMA and mbarrier expect_tx are single-thread instructions.
            if (warp_lane == 0) {
                mbar_wait_parity(&empty_bar[stage], empty_par[stage]);
                empty_par[stage] ^= 1u;
                const int bx = t % tiles_x;
                const int by = t / tiles_x;
                mbar_expect_tx(&full_bar[stage], (unsigned)TILE_BYTES);
                tma_load_2d(&smem_tile[stage].pix[0], &tmap, bx * 16, by * 16,
                            &full_bar[stage]);
            }
            __syncwarp();
        }
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    const int tx = tid & 15;
    const int ty = tid >> 4;
    half *a_row = &A[warp_id][warp_lane * LDA];
    unsigned full_par[TMA_STAGES] = {0, 0};

    int i = 0;
    for (int t = (int)blockIdx.x; t < ntiles; t += stride, ++i) {
        const int stage = i & 1;
        mbar_wait_parity(&full_bar[stage], full_par[stage]);
        full_par[stage] ^= 1u;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        const uint *tile = smem_tile[stage].pix;
        const int bx = t % tiles_x;
        const int by = t / tiles_x;
        const int x = bx * 16 + tx;
        const int y = by * 16 + ty;

        {
            uint4 pk[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            half *f = reinterpret_cast<half *>(pk);
            pack_rgb(f, 0,  u32_to_rgba(tile[(ty + 0) * TILE_W + (tx + 0)]), 0);
            pack_rgb(f, 3,  u32_to_rgba(tile[(ty + 0) * TILE_W + (tx + 2)]), 3);
            pack_rgb(f, 6,  u32_to_rgba(tile[(ty + 0) * TILE_W + (tx + 4)]), 6);
            pack_rgb(f, 9,  u32_to_rgba(tile[(ty + 2) * TILE_W + (tx + 0)]), 9);
            pack_rgb(f, 12, u32_to_rgba(tile[(ty + 2) * TILE_W + (tx + 2)]), 12);
            st16(a_row, pk[0], pk[1]);
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, (const __half *)A[warp_id], LDA);
        wmma::load_matrix_sync(b_frag, (const __half *)&weight_1_shared[0], 8);
        wmma::load_matrix_sync(c_frag, (const half *)&bias_1_shared, 32,
                               wmma::mem_col_major);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        {
            uint4 pk[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
            half *f = reinterpret_cast<half *>(pk);
            pack_rgb(f, 0, u32_to_rgba(tile[(ty + 2) * TILE_W + (tx + 4)]), 16);
            pack_rgb(f, 3, u32_to_rgba(tile[(ty + 4) * TILE_W + (tx + 0)]), 19);
            pack_rgb(f, 6, u32_to_rgba(tile[(ty + 4) * TILE_W + (tx + 2)]), 22);
            pack_rgb(f, 9, u32_to_rgba(tile[(ty + 4) * TILE_W + (tx + 4)]), 25);
            st16(a_row, pk[0], pk[1]);
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, (const __half *)A[warp_id], LDA);
        wmma::load_matrix_sync(b_frag, (const __half *)&weight_1_shared[1], 8);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        for (int k = 0; k < c_frag.num_elements; k++)
            c_frag.x[k] = relu(c_frag.x[k]);
        wmma::store_matrix_sync((half *)A[warp_id], c_frag, LDA,
                                wmma::mem_row_major);

        wmma::load_matrix_sync(a_frag, (const __half *)A[warp_id], LDA);
        wmma::load_matrix_sync(b_frag, (const __half *)&weight_2_shared, 8);
        wmma::load_matrix_sync(c_frag, (const half *)&bias_2_shared, 32,
                               wmma::mem_col_major);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync((half *)A[warp_id], c_frag, 8,
                                wmma::mem_row_major);
        __syncwarp();

        if (x < w && y < h) {
            float4 output;
            output.x = A[warp_id][warp_lane * 8 + 0];
            output.y = A[warp_id][warp_lane * 8 + 1];
            output.z = A[warp_id][warp_lane * 8 + 2];
            od[y * w + x] = rgbaFloatToInt(output);
        }

        if (warp_lane == 0) {
            mbar_arrive(&empty_bar[stage]);
        }
    }
#else
    (void)od;
    (void)w;
    (void)h;
    (void)tmap;
#endif
}

static void encode_tma_map(void *ptr, int width, int height, size_t pitch_bytes)
{
    cuuint64_t dim[2] = {(cuuint64_t)width, (cuuint64_t)height};
    cuuint64_t stride[1] = {(cuuint64_t)pitch_bytes};
    cuuint32_t box[2] = {(cuuint32_t)TILE_W, (cuuint32_t)TILE_H};
    cuuint32_t elem_stride[2] = {1, 1};
    checkCuErrors(cuTensorMapEncodeTiled(
        &g_tmap, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, ptr, dim, stride, box,
        elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

static void build_padded_and_tma(int width, int height, const uint *src,
                                 size_t src_pitch)
{
    const int tiles_x = (width + 15) / 16;
    const int tiles_y = (height + 15) / 16;
    padded_w = (tiles_x - 1) * 16 + TILE_W;
    padded_h = (tiles_y - 1) * 16 + TILE_H;
    if (dPadded) {
        checkCudaErrors(cudaFree(dPadded));
        dPadded = NULL;
    }
    checkCudaErrors(cudaMallocPitch((void **)&dPadded, &padded_pitch,
                                    (size_t)padded_w * sizeof(uint),
                                    padded_h));
    dim3 b(16, 16);
    dim3 g((padded_w + 15) / 16, (padded_h + 15) / 16);
    pad_clamp_kernel<<<g, b>>>(src, width, height, src_pitch, dPadded, padded_w,
                               padded_h, padded_pitch);
    checkCudaErrors(cudaGetLastError());
    encode_tma_map(dPadded, padded_w, padded_h, padded_pitch);
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

   cudaDeviceProp prop;
   checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
   g_sm_major = prop.major;
   if (g_sm_major >= 9) {
       build_padded_and_tma(width, height, dImage, pitch);
       printf("Hopper TMA path: padded %d x %d, warp-specialized persistent CTA\n",
              padded_w, padded_h);
   }
}

extern "C"
void freeTextures()
{
    checkCudaErrors(unbindTex(&rgbaTex));
    checkCudaErrors(cudaFree(dImage));
    checkCudaErrors(cudaFree(dTemp));
    if (dPadded) {
        checkCudaErrors(cudaFree(dPadded));
        dPadded = NULL;
    }
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
                           float e_d, int radius, int iterations)
{
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    const int hopper = (g_sm_major >= 9);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    if (!hopper) {
        checkCudaErrors(bindNormalizedUchar4Tex(&rgbaTex, dImage, desc, width,
                                                height, pitch));
    }

    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    const int tiles_x = (width + 15) / 16;
    const int tiles_y = (height + 15) / 16;
    const int ntiles = tiles_x * tiles_y;
    int grid = prop.multiProcessorCount * 8;
    if (grid > ntiles) {
        grid = ntiles;
    }
    if (grid < 1) {
        grid = 1;
    }

    float totalMs = 0.f;
    for (int i=0; i<iterations; i++)
    {
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaEventRecord(start));

        if (hopper) {
            d_bilateral_filter_hopper<<<grid, HOPPER_THREADS>>>(dDest, width,
                                                                height, g_tmap);
        } else {
            dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
            dim3 blockSize(16, 16);
            d_bilateral_filter<<< gridSize, blockSize>>>(
                dDest, width, height, e_d, radius, rgbaTex);
        }

        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float ms = 0.f;
        checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;

        if (iterations > 1)
        {
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width,
                                         sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
            if (hopper) {
                build_padded_and_tma(width, height, dTemp, pitch);
            } else {
                checkCudaErrors(bindNormalizedUchar4Tex(&rgbaTex, dTemp, desc,
                                                        width, height, pitch));
            }
        }
    }

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    return (totalMs / 1000.0) / (double)iterations;
}
