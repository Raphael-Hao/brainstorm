
#include "gather_64_512.cuh"

#include <algorithm>
#include <brt/common/cuda_utils.h>
#include <cmath>
#include <cublas_v2.h>
#include <random>

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long

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

__global__ void __launch_bounds__(16)
    expert_batch_matmul(float* __restrict__ placeholder,
                        float* __restrict__ placeholder1[64],
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
        ((float4*)((placeholder1[((((int)blockIdx.x) >> 7) * 4)]) +
                   ((((((((int)blockIdx.x) & 127) * 4096)) +
                      ((((int)threadIdx.x) >> 2) * 1024)) +
                     (k_outer_outer * 16)) +
                    ((((int)threadIdx.x) & 3) * 4))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 64))))[0] =
        ((float4*)(placeholder1[(((((int)blockIdx.x) >> 7) * 4) + 1)] +
                   ((((((((int)blockIdx.x) & 127) * 4096) +
                       ((((int)threadIdx.x) >> 2) * 1024)) +
                      (k_outer_outer * 16)) +
                     ((((int)threadIdx.x) & 3) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] =
        ((float4*)(placeholder1[(((((int)blockIdx.x) >> 7) * 4) + 2)] +
                   ((((((((int)blockIdx.x) & 127) * 4096) +
                       ((((int)threadIdx.x) >> 2) * 1024)) +
                      (k_outer_outer * 16)) +
                     ((((int)threadIdx.x) & 3) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 192))))[0] =
        ((float4*)(placeholder1[(((((int)blockIdx.x) >> 7) * 4) + 3)] +
                   ((((((((int)blockIdx.x) & 127) * 4096) +
                       ((((int)threadIdx.x) >> 2) * 1024)) +
                      (k_outer_outer * 16)) +
                     ((((int)threadIdx.x) & 3) * 4)))))[0];
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

__global__ void __launch_bounds__(32)
    pointer_array_assign(float** dst, float* src, int index[]) {
  dst[((blockIdx.x * 32) + threadIdx.x)] =
      src + index[((blockIdx.x * 32) + threadIdx.x)] * 1024 * 512;
}

__global__ void __launch_bounds__(1024)
    array_value_comp(float** dst, float* src) {
  if (dst[blockIdx.y][blockIdx.x * 1024 + threadIdx.x] !=
      src[blockIdx.y * 524288 + blockIdx.x * 1024 + threadIdx.x]) {
    printf("%d-th value not equal:  %f, %f\n",
           blockIdx.y * 524288 + blockIdx.x * 1024 + threadIdx.x,
           dst[blockIdx.y][blockIdx.x * 1024 + threadIdx.x],
           src[blockIdx.y * 524288 + blockIdx.x * 1024 + threadIdx.x]);
  }
  __syncthreads();
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

void generate_expert_index(int* index, int size, int expert_num) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, expert_num - 1);
  for (int i = 0; i < size; ++i) {
    index[i] = dis(gen);
  }
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
  cublasHandle_t cublas_handle;
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

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

  int* A_indices;
  CUDA_CHECK(cudaMallocHost((void**)&A_indices, sizeof(int) * size_A));
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 512; ++j) {
      A_indices[i * 512 + j] = A_row_indices[i];
    }
  }

  int* A_indices_d;
  CUDA_CHECK(cudaMalloc((void**)&A_indices_d, sizeof(int) * size_A));
  CUDA_CHECK(cudaMemcpy(A_indices_d, A_indices, sizeof(int) * size_A,
                        cudaMemcpyHostToDevice));

  float *A_h, *B_h, *C_h, *C_h_grand;
  CUDA_CHECK(cudaMallocHost((void**)&A_h, size_A * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void**)&B_h, size_B * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void**)&C_h, size_C * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void**)&C_h_grand, size_C * sizeof(float)));

  float *A_d, *A_dispatch_d, *B_d, *B_d_linear, *C_d, *C_d_grand;
  float** B_d_array;
  CUDA_CHECK(cudaMalloc((void**)&A_d, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&A_dispatch_d, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&B_d, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&B_d_linear, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&B_d_array, sizeof(float*) * 64));
  CUDA_CHECK(cudaMalloc((void**)&C_d, size_C * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&C_d_grand, size_C * sizeof(float)));

  init_with_val(A_h, size_A, 1.0f);
  init_with_rand(B_h, size_B);
  init_with_val(C_h, size_C, 0.0f);

  CUDA_CHECK(
      cudaMemcpy(A_d, A_h, size_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(B_d, B_h, size_B * sizeof(float), cudaMemcpyHostToDevice));

  int* expert_indexes;
  CUDA_CHECK(cudaMallocHost((void**)&expert_indexes, sizeof(int) * 64));
  generate_expert_index(expert_indexes, 64, 64);

  for (auto i = 0; i < 64; ++i) {
    CUDA_CHECK(cudaMemcpy(
        B_d_linear + i * 1024 * 512, B_d + expert_indexes[i] * 1024 * 512,
        1024 * 512 * sizeof(float), cudaMemcpyDeviceToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  pointer_array_assign<<<2, 32, 0, stream>>>(B_d_array, B_d, expert_indexes);
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 grids(512, 64);
  dim3 blocks(1024, 1);

  array_value_comp<<<grids, blocks, 0, stream>>>(B_d_array, B_d_linear);
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 gather_grid(16, 16);
  dim3 gather_block(32, 4);
  gather_kernel_kernel_16_16_32_4<<<gather_grid, gather_block, 0, stream>>>(
      A_indices_d, A_dispatch_d, A_d);
  CUDA_CHECK(cudaDeviceSynchronize());
  dim3 threads_per_block(4 * 4);
  dim3 blocks_per_grid(512 * 64 / threads_per_block.x);
  float alpha = 1.0f;
  float beta = 0.0f;
  // warm up
  for (auto i = 0; i < 1024; ++i) {
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 64, 512,
                             1024, &alpha, A_d, 1024, B_d_linear, 512, &beta,
                             C_d, 512));
    default_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d_linear, C_d_grand);
    expert_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d_array, C_d);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // start timing
  int test_num = 1000;
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (auto i = 0; i < test_num; ++i) {
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 64, 512,
                             1024, &alpha, A_dispatch_d, 1024, B_d_linear, 512,
                             &beta, C_d, 512));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("default cublas matmul without gather %f ms\n",
         elapsed_time / test_num);

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (auto i = 0; i < test_num; ++i) {
    gather_kernel_kernel_16_16_32_4<<<gather_grid, gather_block, 0, stream>>>(
        A_indices_d, A_dispatch_d, A_d);
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 64, 512,
                             1024, &alpha, A_dispatch_d, 1024, B_d_linear, 512,
                             &beta, C_d, 512));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("default cublas matmul with gather %f ms\n", elapsed_time / test_num);

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (auto i = 0; i < test_num; ++i) {
    default_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d_linear, C_d_grand);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("default batch matmul %f ms\n", elapsed_time / test_num);

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (auto i = 0; i < test_num; ++i) {
    expert_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d_array, C_d);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("expert batch matmul without pointer gather %f ms\n",
         elapsed_time / test_num);

  CUDA_CHECK(cudaEventRecord(start, stream));
  for (auto i = 0; i < test_num; ++i) {
    pointer_array_assign<<<2, 32, 0, stream>>>(B_d_array, B_d, expert_indexes);
    expert_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        A_d, B_d_array, C_d);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("expert batch matmul with pointer gather %f ms\n",
         elapsed_time / test_num);

  CUDA_CHECK(cudaMemcpy(C_h_grand, C_d_grand, size_C * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(C_h, C_d, size_C * sizeof(float), cudaMemcpyDeviceToHost));
  bool if_equal = check_results(C_h_grand, C_h, size_C);
  if (if_equal) {
    printf("passed\n");
  } else {
    printf("failed\n");
  }
  return 0;
}
