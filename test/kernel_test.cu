/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <algorithm>
#include <brt/runtime/cuda_utils.h>
#include <cuda_runtime.h>
#include <random>

extern "C" {
__global__ void __launch_bounds__(16)
    default_batch_matmul(float* __restrict__ placeholder,
                         float* __restrict__ placeholder1,
                         float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[1];
  __shared__ float placeholder_d_shared[64];
  __shared__ float placeholder_shared[256];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(placeholder_d_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 7) * 4096) +
                                     ((((int)threadIdx.x) >> 2) * 1024)) +
                                    (k_outer_outer * 16)) +
                                   ((((int)threadIdx.x) & 3) * 4)))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 2097152) +
                                       ((((int)blockIdx.x) & 127) * 4096)) +
                                      ((((int)threadIdx.x) >> 2) * 1024)) +
                                     (k_outer_outer * 16)) +
                                    ((((int)threadIdx.x) & 3) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 64))))[0] =
        ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 7) * 2097152) +
                                        ((((int)blockIdx.x) & 127) * 4096)) +
                                       ((((int)threadIdx.x) >> 2) * 1024)) +
                                      (k_outer_outer * 16)) +
                                     ((((int)threadIdx.x) & 3) * 4)) +
                                    524288))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] =
        ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 7) * 2097152) +
                                        ((((int)blockIdx.x) & 127) * 4096)) +
                                       ((((int)threadIdx.x) >> 2) * 1024)) +
                                      (k_outer_outer * 16)) +
                                     ((((int)threadIdx.x) & 3) * 4)) +
                                    1048576))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 192))))[0] =
        ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 7) * 2097152) +
                                        ((((int)blockIdx.x) & 127) * 4096)) +
                                       ((((int)threadIdx.x) >> 2) * 1024)) +
                                      (k_outer_outer * 16)) +
                                     ((((int)threadIdx.x) & 3) * 4)) +
                                    1572864))))[0];
    __syncthreads();
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(((((int)threadIdx.x) >> 2) * 16))] *
          placeholder_shared[((((int)threadIdx.x) * 16))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 1))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 1))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 2))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 2))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 3))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 3))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 4))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 4))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 5))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 5))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 6))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 6))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 7))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 7))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 8))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 8))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 9))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 9))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 10))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 10))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 11))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 11))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 12))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 12))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 13))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 13))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 14))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 14))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[((((((int)threadIdx.x) >> 2) * 16) + 15))] *
          placeholder_shared[(((((int)threadIdx.x) * 16) + 15))]));
  }
  T_batch_matmul_NT[((
      ((((((int)blockIdx.x) >> 7) * 2048) + ((((int)threadIdx.x) >> 2) * 512)) +
       ((((int)blockIdx.x) & 127) * 4)) +
      (((int)threadIdx.x) & 3)))] = T_batch_matmul_NT_local[(0)];
}

__global__ void __launch_bounds__(128)
    matmul_1_1024_512_1(float* __restrict__ placeholder,
                        float* __restrict__ placeholder1,
                        float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[1];
  __shared__ float placeholder_d_shared[512];
  __shared__ float placeholder_shared[8192];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
#pragma unroll
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(placeholder_d_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)(placeholder + (k_outer_outer * 512) +
                   ((((((int)blockIdx.x) >> 2) * 1024) +
                     (((int)threadIdx.x) * 4)))))[0];
#pragma unroll
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        ((float4*)(placeholder_shared +
                   ((((int)threadIdx.x) * 4 + k * 512))))[0] =
            ((float4*)(placeholder1 +
                       (((((((((int)blockIdx.x) >> 2) * 524288) +
                            ((((int)blockIdx.x) & 3) * 131072))) +
                          (k_outer_outer * 512) + (k_outer * 64) + (k * 8192)) +
                         ((((int)threadIdx.x) >> 4) * 1024) +
                         ((((int)threadIdx.x) & 15) * 4)))))[0];
      }
      __syncthreads();
#pragma unroll
      for (int k = 0; k < 64; ++k) {
        T_batch_matmul_NT_local[(0)] +=
            placeholder_d_shared[k + k_outer * 64] *
            placeholder_shared[((((int)threadIdx.x) * 64) + k)];
      }
    }
  }
  T_batch_matmul_NT[(
      ((((((int)blockIdx.x) >> 2) * 512) + ((((int)blockIdx.x) & 3) * 128)) +
       ((int)threadIdx.x)))] = T_batch_matmul_NT_local[(0)];
}

__global__ void __launch_bounds__(16)
    matmul_1_1024_512(float* __restrict__ placeholder,
                      float* __restrict__ placeholder1,
                      float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[1];
  __shared__ float placeholder_d_shared[64];
  __shared__ float placeholder_shared[1024];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    ((float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2))))[0] = ((
        float2*)(placeholder +
                 (((((((int)blockIdx.x) >> 5) * 1024) + (k_outer_outer * 64)) +
                   (((int)threadIdx.x) * 2)))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 32))))[0] =
        ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 1024) +
                                     (k_outer_outer * 64)) +
                                    (((int)threadIdx.x) * 2)) +
                                   32))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)(placeholder1 +
                   ((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                     (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 64))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     1024))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 192))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     3072))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 320))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     5120))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 448))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     7168))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     8192))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 576))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     9216))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     10240))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 704))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     11264))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     12288))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 832))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     13312))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     14336))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 960))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 64)) +
                      (((int)threadIdx.x) * 4)) +
                     15360))))[0];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 32; ++k_inner) {
        T_batch_matmul_NT_local[(0)] =
            (T_batch_matmul_NT_local[(0)] +
             (placeholder_d_shared[(((k_outer_inner * 32) + k_inner))] *
              placeholder_shared[(
                  (((((int)threadIdx.x) * 64) + (k_outer_inner * 32)) +
                   k_inner))]));
      }
    }
  }
  T_batch_matmul_NT[(((((int)blockIdx.x) * 16) + ((int)threadIdx.x)))] =
      T_batch_matmul_NT_local[(0)];
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
  brt::CUDA_CHECK(cudaSetDevice(device_id));
  // create stream
  cudaStream_t stream;
  brt::CUDA_CHECK(cudaStreamCreate(&stream));

  // create CUDA events
  cudaEvent_t start, stop;
  brt::CUDA_CHECK(cudaEventCreate(&start));
  brt::CUDA_CHECK(cudaEventCreate(&stop));
  float elapsed_time;

  int A_row_indices[64];
  for (int i = 0; i < 64; ++i) {
    A_row_indices[i] = i;
  }
  std::random_shuffle(A_row_indices, A_row_indices + 64);

  float *A_h, *B_h, *C_h, *C_h_grand;
  brt::CUDA_CHECK(cudaMallocHost((void**)&A_h, size_A * sizeof(float)));
  brt::CUDA_CHECK(cudaMallocHost((void**)&B_h, size_B * sizeof(float)));
  brt::CUDA_CHECK(cudaMallocHost((void**)&C_h, size_C * sizeof(float)));
  brt::CUDA_CHECK(cudaMallocHost((void**)&C_h_grand, size_C * sizeof(float)));

  float *A_d, *B_d, *C_d, *C_d_grand;
  brt::CUDA_CHECK(cudaMalloc((void**)&A_d, size_A * sizeof(float)));
  brt::CUDA_CHECK(cudaMalloc((void**)&B_d, size_B * sizeof(float)));
  brt::CUDA_CHECK(cudaMalloc((void**)&C_d, size_C * sizeof(float)));
  brt::CUDA_CHECK(cudaMalloc((void**)&C_d_grand, size_C * sizeof(float)));

  init_with_val(A_h, size_A, 1.0f);
  init_with_rand(B_h, size_B);

  brt::CUDA_CHECK(
      cudaMemcpy(A_d, A_h, size_A * sizeof(float), cudaMemcpyHostToDevice));
  brt::CUDA_CHECK(
      cudaMemcpy(B_d, B_h, size_B * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads_per_block(4 * 4);
  dim3 blocks_per_grid(512 * 64 / threads_per_block.x);
  dim3 matmul_block_size(16);   //(128);
  dim3 matmul_grid_size(2048);  //(256);

  // warm up
  for (auto i = 0; i < 1024; ++i) {
    default_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d, C_d_grand);
    matmul_1_1024_512<<<matmul_grid_size, matmul_block_size, 0, stream>>>(
        A_d, B_d, C_d);
  }
  brt::CUDA_CHECK(cudaDeviceSynchronize());

  // start timing
  int test_num = 1000;

  brt::CUDA_CHECK(cudaEventRecord(start, stream));
  for (auto i = 0; i < test_num; ++i) {
    default_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d, C_d_grand);
  }
  brt::CUDA_CHECK(cudaEventRecord(stop, stream));
  brt::CUDA_CHECK(cudaEventSynchronize(stop));
  brt::CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("default batch matmul %f ms\n", elapsed_time / test_num);

  brt::CUDA_CHECK(cudaEventRecord(start, stream));
  for (auto i = 0; i < test_num; ++i) {
    matmul_1_1024_512<<<matmul_grid_size, matmul_block_size, 0, stream>>>(
        A_d, B_d, C_d);
  }
  brt::CUDA_CHECK(cudaEventRecord(stop, stream));
  brt::CUDA_CHECK(cudaEventSynchronize(stop));
  brt::CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("expert batch matmul without pointer gather %f ms\n",
         elapsed_time / test_num);

  brt::CUDA_CHECK(cudaMemcpy(C_h_grand, C_d_grand, size_C * sizeof(float),
                        cudaMemcpyDeviceToHost));
  brt::CUDA_CHECK(
      cudaMemcpy(C_h, C_d, size_C * sizeof(float), cudaMemcpyDeviceToHost));
  bool if_equal = check_results(C_h_grand, C_h, size_C);
  if (if_equal) {
    printf("passed\n");
  } else {
    printf("failed\n");
  }
  return 0;
}
