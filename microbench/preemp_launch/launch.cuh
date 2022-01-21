#pragma once

#include <cuda_runtime.h>

#define LAUCH_CHECK(launch_flag, x) \
  if (launch_flag != x) {           \
    return                          \
  }

#define CUDA_CHECK(x)                                   \
  do {                                                  \
    cudaError_t error = x;                              \
    if (error != cudaSuccess) {                         \
      printf("Error: %s\n", cudaGetErrorString(error)); \
      exit(1);                                          \
    }                                                   \
  } while (0)
