#pragma once
#include <cuda_runtime.h>

#include "launch.cuh"

template <int F>
void __global__ kernel_add(float *a, float *b, float *c, int n,
                           int launch_flag) {
  LAUCH_CHECK(launch_flag, F);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
