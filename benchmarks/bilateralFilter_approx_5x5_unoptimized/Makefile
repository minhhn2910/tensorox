# Standalone CUDA bilateral filter. Requires nvcc (CUDA toolkit).
# No NVIDIA samples, OpenGL, or GLUT.
#
# In this repo:  make
# Copied elsewhere: make   (uses nvcc -arch=native)
# Override: make CUDA_PATH=/usr/local/cuda SMS=86

TARGET := $(notdir $(CURDIR))
SRCS := bilateralFilter.cpp bmploader.cpp bilateral_kernel.cu
NVCCFLAGS ?= -O3 -std=c++14

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CUDA_MK := $(abspath $(THIS_DIR)/../../scripts/cuda.mk)

ifneq ($(wildcard $(CUDA_MK)),)
  include $(CUDA_MK)
  ARCH_FLAGS := $(GENCODE_FLAGS)
else
  NVCC ?= nvcc
  ifdef CUDA_PATH
    NVCC := $(CUDA_PATH)/bin/nvcc
  endif
  ARCH_FLAGS ?= -arch=native
endif

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $(SRCS) -o $@

run: $(TARGET)
	./$(TARGET) -benchmark

clean:
	rm -f $(TARGET) *.o nature_*.ppm
