#pragma once

#include <cuda_runtime.h>
#include <string>

#define LAUNCH_CHECK(launch_flag, kFlag) \
  if (*launch_flag != kFlag) {           \
    return;                              \
  }

#define LAUNCH_CHECK_ASM(launch_flag, kFlag) \
  if (*launch_flag != kFlag) {               \
    asm("trap;");                            \
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