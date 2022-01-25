#pragma once
#include <cuda_runtime.h>

#include "launch.cuh"

void __global__ set_flag(int *kFlag, int kValue) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 1) {
    *kFlag = kValue;
  }
}

template <int kFlag>
void __global__ kernel_add(float *a, float *b, float *c, int n,
                           int *launch_flag) {
  LAUCH_CHECK(launch_flag, kFlag);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
