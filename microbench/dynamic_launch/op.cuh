#pragma once
#include "launch.cuh"

#include <cuda_runtime.h>

void __global__ set_flag(int *kFlag, int kValue) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 0) {
    *kFlag = kValue;
  }
}

void __global__ simple_add(float *a, float *b, float *c, int n,
                           int *launch_flag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("simple_add: %d\n", idx);
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

template <int kFlag>
void __global__ dynamic_add(float *a, float *b, float *c, int n,
                            int *launch_flag) {
  LAUNCH_CHECK(launch_flag, kFlag);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    printf("launch_flag = %d\n", kFlag);
  }
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void helloFromGpu() { printf("Hello from GPU.\n"); }