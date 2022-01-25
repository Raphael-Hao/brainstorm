#pragma once

#include <cuda_runtime.h>

#include <string>

#define LAUCH_CHECK(launch_flag, kFlag)                      \
  if (*launch_flag != kFlag) {                               \
    return;                                                  \
  } else {                                                   \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;         \
    if (idx == 1) {                                          \
      printf("Launch the %d-th candidate kernel!\n", kFlag); \
    }                                                        \
  }

#define CUDA_CHECK(x) __CUDA_CHECK(x, __FILE__, __LINE__)

inline void __CUDA_CHECK(cudaError_t x, const char *file, int line) {
  do {
    if (x != cudaSuccess) {
      fprintf(stderr, "Error: %s, from file <%s>, line %i.\n",
              cudaGetErrorString(x), file, line);
      exit(1);
    }
  } while (0);
}