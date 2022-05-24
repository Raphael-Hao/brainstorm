/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

// extract to string

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define thread_num 1024
#define expert_num 1024
#define num_samples 1024
extern "C" __global__ void cumsum(int* input0 /* (num_samples, expert_num) */,
                                  int* output0 /* (num_samples, expert_num) */) {
  // [thread_extent] blockIdx.x = expert_num
  // [thread_extent] threadIdx.x = 1024
  __shared__ int temp[thread_num + 1];
  int thid = threadIdx.x, bid = blockIdx.x;
  int last_sum = -1;

  for (int S = 0; S < num_samples; S += thread_num) {
    int offset = 1;
    if (S + thid < num_samples) {
      temp[thid] = input0[thid * expert_num + bid];
    }
    // sum all temp[0:thid] to temp[thread_num - 1]
    for (int d = thread_num >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (thid < d) {
        temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
      }
      offset *= 2;
    }
    // store the sum of temp[0:thid] to temp[thread_num] and put temp[thread_num - 1] to 0
    if (thid == 0) {
      temp[thread_num] = temp[thread_num - 1];
      temp[thread_num - 1] = 0;
    }
    // reverse dispatch the sum of temp[0:thid] to temp[thid+1]
    for (int d = 1; d < thread_num; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        int t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    __syncthreads();
    if (S + thid < num_samples) {
      output0[thid * expert_num + bid] = temp[thid + 1] + last_sum;
    }
    __syncthreads();
    last_sum += temp[thread_num];
    output0 += thread_num * expert_num;
    input0 += thread_num * expert_num;
  }
}

#define __dtype float

/*
samples = n x hidden
expert_num =16 x capacity = 16 x hidden
*/
extern "C" __global__ __launch_bounds__(1024) void execute(__dtype* __restrict__ gates1_s,
                                                           int* __restrict__ indices1_s,
                                                           int* __restrict__ locations1_s,
                                                           __dtype* __restrict__ reshaped_input,
                                                           __dtype* __restrict__ dispatched_input,
                                                           int samples, int hidden, int capacity) {
  // [thread_extent] blockIdx.x = 512
  // [thread_extent] threadIdx.x = 1024

  for (int i = blockIdx.x; i < samples; i += gridDim.x)
    if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
#pragma unroll
      for (int j = threadIdx.x; j < hidden; j += 1024)
        atomicAdd(&dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) * (hidden) + j],
                  gates1_s[i] * reshaped_input[i * (hidden) + j]);
    }
}


__global__ void staticReverse(int* d, int n) {
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n - t - 1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

__global__ void dynamicReverse(int* d, int n) {
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n - t - 1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void) {
  const int n = 64;
  int a[n], r[n], d[n];

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n - i - 1;
    d[i] = 0;
  }

  int* d_d;
  cudaMalloc(&d_d, n * sizeof(int));

  // run version with static shared memory
  cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1, n>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1, n, n * sizeof(int)>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}