/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/runtime/cuda_utils.h>

#include <algorithm>
#include "./matmul.cuh"
#include <random>

extern "C" {
__global__ void __launch_bounds__(32)
    horizontal_matmul(float* __restrict__ placeholder_32, float* __restrict__ placeholder1_32,
                      float* __restrict__ T_batch_matmul_NT_32, float* __restrict__ placeholder_16,
                      float* __restrict__ placeholder1_16, float* __restrict__ T_batch_matmul_NT_16,
                      float* __restrict__ placeholder_8, float* __restrict__ placeholder1_8,
                      float* __restrict__ T_batch_matmul_NT_8, float* __restrict__ placeholder_4,
                      float* __restrict__ placeholder1_4, float* __restrict__ T_batch_matmul_NT_4,
                      float* __restrict__ placeholder_2, float* __restrict__ placeholder1_2,
                      float* __restrict__ T_batch_matmul_NT_2, int n_32 = 0, int n_16 = 0,
                      int n_8 = 0, int n_4 = 0, int n_2 = 0) {
  dim3 _blockIdx = blockIdx;
  __shared__ float placeholder_d_shared[2048];
  __shared__ float placeholder_shared[4096];
  if (blockIdx.x < n_32 * 128) {
    matmul_32_1024_512(placeholder_32, placeholder1_32, T_batch_matmul_NT_32, placeholder_d_shared,
                       placeholder_shared, _blockIdx, threadIdx);
  } else if (blockIdx.x < n_32 * 128 + n_16 * 32) {
    _blockIdx.x -= n_32 * 128;
    matmul_16_1024_512(placeholder_16, placeholder1_16, T_batch_matmul_NT_16, placeholder_d_shared,
                       placeholder_shared, _blockIdx, threadIdx);
  } else if (blockIdx.x < n_32 * 128 + n_16 * 32 + n_8 * 64) {
    if (threadIdx.x < 16) {
      _blockIdx.x -= n_32 * 128 + n_16 * 32;
      matmul_8_1024_512(placeholder_8, placeholder1_8, T_batch_matmul_NT_8, placeholder_d_shared,
                        placeholder_shared, _blockIdx, threadIdx);
    }
  } else if (blockIdx.x < n_32 * 128 + n_16 * 32 + n_8 * 64 + n_4 * 32) {
    if (threadIdx.x < 16) {
      _blockIdx.x -= n_32 * 128 + n_16 * 32 + n_8 * 64;
      matmul_4_1024_512(placeholder_4, placeholder1_4, T_batch_matmul_NT_4, placeholder_d_shared,
                        placeholder_shared, _blockIdx, threadIdx);
    }
  } else if (blockIdx.x < n_32 * 128 + n_16 * 32 + n_8 * 64 + n_4 * 32 + n_2 * 16) {
    _blockIdx.x -= n_32 * 128 + n_16 * 32 + n_8 * 64 + n_4 * 32;
    matmul_2_1024_512(placeholder_2, placeholder1_2, T_batch_matmul_NT_2, placeholder_d_shared,
                      placeholder_shared, _blockIdx, threadIdx);
  }
}
}

void init_with_val(float* __restrict__ data, int size, float value) {
  for (int i = 0; i < size; ++i) {
    data[i] = value;
  }
}

void init_with_rand(float* __restrict__ data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = (rand() % 100) / 100.0;
  }
}

bool check_results(float* lfs, float* rhs, int size) {
  for (int i = 0; i < size; ++i) {
    if (std::abs(lfs[i] - rhs[i]) > 1e-5) {
      printf("%d-th value not equal:  %f, %f\n", i, lfs[i], rhs[i]);
      return false;
    }
  }
  return true;
}

int main(int argc, char const* argv[]) {
  size_t size_A = 1024 * 64;
  size_t size_B = 1024 * 512 * 64;
  size_t size_C = 512 * 64;

  int device_id = 0;
  // set device
  printf("setting to device %d\n", device_id);
  CUDA_CHECK(cudaSetDevice(device_id));
  // create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // create CUDA events
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float elapsed_time;

  int A_row_indices[64];
  for (int i = 0; i < 64; ++i) {
    A_row_indices[i] = i;
  }
  std::random_shuffle(A_row_indices, A_row_indices + 64);

  float *A_h, *B_h, *C_h, *C_h_grand;
  CUDA_CHECK(cudaMallocHost((void**)&A_h, size_A * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void**)&B_h, size_B * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void**)&C_h, size_C * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void**)&C_h_grand, size_C * sizeof(float)));

  float *A_d, *B_d, *C_d, *C_d_grand;
  CUDA_CHECK(cudaMalloc((void**)&A_d, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&B_d, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&C_d, size_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&C_d_grand, size_C * sizeof(float)));

  init_with_val(A_h, size_A, 1.0f);
  init_with_rand(B_h, size_B);

  CUDA_CHECK(cudaMemcpy(A_d, A_h, size_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, size_B * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads_per_block(32);
  dim3 blocks_per_grid(128 + 32 + 64 + 32 + 16);
  // dim3 matmul_block_size(16);   //(128);
  // dim3 matmul_grid_size(2048);  //(256);

  // warm up
  for (auto i = 0; i < 1024; ++i) {
    horizontal_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d, C_d_grand, A_d, B_d, C_d_grand, A_d, B_d, C_d_grand, A_d, B_d, C_d_grand, A_d,
        B_d, C_d_grand, 1, 1, 1, 1, 1);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // start timing
  int test_num = 1000;

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (auto i = 0; i < test_num; ++i) {
    horizontal_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d, C_d_grand, A_d, B_d, C_d_grand, A_d, B_d, C_d_grand, A_d, B_d, C_d_grand, A_d,
        B_d, C_d_grand, 1, 1, 1, 1, 1);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("default batch matmul %f ms\n", elapsed_time / test_num);

  CUDA_CHECK(cudaMemcpy(C_h_grand, C_d_grand, size_C * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(C_h, C_d, size_C * sizeof(float), cudaMemcpyDeviceToHost));
  bool if_equal = check_results(C_h_grand, C_h, size_C);
  if (if_equal) {
    printf("passed\n");
  } else {
    printf("failed\n");
  }
  return 0;
}
