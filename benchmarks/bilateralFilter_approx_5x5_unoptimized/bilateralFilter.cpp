#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"

extern "C" void initTexture(int width, int height, unsigned int *hImage);
extern "C" void freeTextures();
extern "C" double bilateralFilterRGBA(unsigned int *d_dest, int width, int height,
                                      float e_d, int radius, int iterations);
extern "C" void updateGaussian(float delta, int radius);
extern "C" void LoadBMPFile(unsigned int **dst, unsigned int *width,
                            unsigned int *height, const char *name);

static int filter_radius = 5;
static int iterations = 1;
static float gaussian_delta = 4.f;
static float euclidean_delta = 0.1f;

static bool has_flag(int argc, char **argv, const char *name) {
  size_t n = strlen(name);
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], name, n) == 0 &&
        (argv[i][n] == '\0' || argv[i][n] == '=')) {
      return true;
    }
  }
  return false;
}

static const char *arg_str(int argc, char **argv, const char *name,
                           const char *defval) {
  size_t n = strlen(name);
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], name, n) == 0 && argv[i][n] == '=') {
      return argv[i] + n + 1;
    }
  }
  return defval;
}

static int arg_int(int argc, char **argv, const char *name, int defval) {
  const char *s = arg_str(argc, argv, name, NULL);
  return s ? atoi(s) : defval;
}

static void save_ppm_rgb(const char *path, const unsigned char *rgba, int w,
                         int h) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "cannot write %s\n", path);
    return;
  }
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  for (int i = 0; i < w * h; i++) {
    fputc(rgba[i * 4 + 0], f);
    fputc(rgba[i * 4 + 1], f);
    fputc(rgba[i * 4 + 2], f);
  }
  fclose(f);
  printf("Wrote %s\n", path);
}

static unsigned int *make_synthetic(unsigned int *w, unsigned int *h) {
  *w = 1024;
  *h = 1024;
  unsigned int n = (*w) * (*h);
  unsigned int *img = (unsigned int *)malloc(n * sizeof(unsigned int));
  for (unsigned int y = 0; y < *h; y++) {
    for (unsigned int x = 0; x < *w; x++) {
      unsigned r = (x * 255u) / (*w);
      unsigned g = (y * 255u) / (*h);
      unsigned b = ((x ^ y) * 255u) / ((*w) > 0 ? *w : 1);
      img[y * (*w) + x] = r | (g << 8) | (b << 16);
    }
  }
  printf("No BMP given; using synthetic %u x %u image\n", *w, *h);
  return img;
}

static unsigned int *load_image(const char *path, unsigned int *w,
                                unsigned int *h) {
  if (!path) {
    return make_synthetic(w, h);
  }
  unsigned int *img = NULL;
  LoadBMPFile(&img, w, h, path);
  if (!img) {
    fprintf(stderr, "failed to load %s\n", path);
    exit(EXIT_FAILURE);
  }
  printf("Loaded '%s', %u x %u\n", path, *w, *h);
  return img;
}

static int file_exists(const char *p) {
  FILE *f = fopen(p, "rb");
  if (!f) {
    return 0;
  }
  fclose(f);
  return 1;
}

static const char *find_default_image(void) {
  static const char *cands[] = {
      "data/1024nature.bmp", "data/nature_monte.bmp", "1024nature.bmp",
      "nature_monte.bmp", NULL};
  for (int i = 0; cands[i]; i++) {
    if (file_exists(cands[i])) {
      return cands[i];
    }
  }
  return NULL;
}

static void print_help(const char *argv0) {
  printf("Usage: %s [options]\n", argv0);
  printf("  -image=path.bmp   input 24-bit BMP (optional; else data/ or synthetic)\n");
  printf("  -radius=n         filter radius (default %d)\n", filter_radius);
  printf("  -passes=n         iterations (default %d)\n", iterations);
  printf("  -benchmark        time 150 cycles, print throughput\n");
  printf("  -out=file.ppm     write RGB result\n");
}

int main(int argc, char **argv) {
  if (has_flag(argc, argv, "-h") || has_flag(argc, argv, "--help")) {
    print_help(argv[0]);
    return 0;
  }

  filter_radius = arg_int(argc, argv, "-radius", filter_radius);
  iterations = arg_int(argc, argv, "-passes", iterations);
  const char *image_path = arg_str(argc, argv, "-image", NULL);
  const char *out_path = arg_str(argc, argv, "-out", NULL);
  int do_bench = has_flag(argc, argv, "-benchmark");

  if (!image_path) {
    image_path = find_default_image();
  }

  unsigned int width = 0, height = 0;
  unsigned int *hImage = load_image(image_path, &width, &height);

  int dev = 0;
  checkCudaErrors(cudaSetDevice(dev));
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, dev));
  printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  updateGaussian(gaussian_delta, filter_radius);
  initTexture((int)width, (int)height, hImage);

  unsigned int *dResult = NULL;
  size_t pitch = 0;
  checkCudaErrors(cudaMallocPitch((void **)&dResult, &pitch,
                                  width * sizeof(unsigned int), height));

  // warmup
  bilateralFilterRGBA(dResult, (int)width, (int)height, euclidean_delta,
                      filter_radius, iterations);
  checkCudaErrors(cudaDeviceSynchronize());

  const int cycles = do_bench ? 150 : 1;
  double tsum = 0.0;
  printf("Running %d cycle(s), radius=%d, passes=%d\n", cycles, filter_radius,
         iterations);
  for (int i = 0; i < cycles; i++) {
    tsum += bilateralFilterRGBA(dResult, (int)width, (int)height,
                                euclidean_delta, filter_radius, iterations);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  double t = tsum / (double)cycles;
  printf("bilateralFilter, Throughput = %.4f M RGBA Pixels/s, Time = %.5f s, "
         "Size = %u RGBA Pixels\n",
         (1.0e-6 * width * height) / t, t, width * height);

  if (out_path || !do_bench) {
    unsigned int *hResult =
        (unsigned int *)malloc(width * height * sizeof(unsigned int));
    checkCudaErrors(cudaMemcpy2D(hResult, sizeof(unsigned int) * width, dResult,
                                 pitch, sizeof(unsigned int) * width, height,
                                 cudaMemcpyDeviceToHost));
    char defout[64];
    snprintf(defout, sizeof(defout), "nature_%02d.ppm", filter_radius);
    save_ppm_rgb(out_path ? out_path : defout, (unsigned char *)hResult,
                 (int)width, (int)height);
    free(hResult);
  }

  checkCudaErrors(cudaFree(dResult));
  freeTextures();
  free(hImage);
  return 0;
}
