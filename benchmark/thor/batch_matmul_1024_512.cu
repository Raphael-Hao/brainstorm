
#include <brainstorm/common/cuda_utils.h>

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
        ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 7) * 1) +
                                     ((((int)threadIdx.x) >> 2) * 1024)) +
                                    (k_outer_outer * 16)) +
                                   ((((int)threadIdx.x) & 3) * 4)))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)((placeholder1[(((int)blockIdx.x) >> 7)]) +
                   ((((((((int)blockIdx.x) & 127) * 4096)) +
                      ((((int)threadIdx.x) >> 2) * 1024)) +
                     (k_outer_outer * 16)) +
                    ((((int)threadIdx.x) & 3) * 4))))[0];
    // ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 64))))[0] =
    //     ((float4*)(placeholder1[((((int)blockIdx.x) >> 7) + 1)] +
    //                ((((((((int)blockIdx.x) & 127) * 4096) +
    //                    ((((int)threadIdx.x) >> 2) * 1024)) +
    //                   (k_outer_outer * 16)) +
    //                  ((((int)threadIdx.x) & 3) * 4)))))[0];
    // ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] =
    //     ((float4*)(placeholder1[((((int)blockIdx.x) >> 7) + 2)] +
    //                ((((((((int)blockIdx.x) & 127) * 4096) +
    //                    ((((int)threadIdx.x) >> 2) * 1024)) +
    //                   (k_outer_outer * 16)) +
    //                  ((((int)threadIdx.x) & 3) * 4)))))[0];
    // ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 192))))[0] =
    //     ((float4*)(placeholder1[((((int)blockIdx.x) >> 7) + 3)] +
    //                ((((((((int)blockIdx.x) & 127) * 4096) +
    //                    ((((int)threadIdx.x) >> 2) * 1024)) +
    //                   (k_outer_outer * 16)) +
    //                  ((((int)threadIdx.x) & 3) * 4)))))[0];
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
}

void init_with_val(float* data, int size, float value) {
  for (int i = 0; i < size; ++i) {
    data[i] = value;
  }
}

void init_with_rand(float* data, int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = (rand() % 100) / 100.0;
  }
}

bool check_results(float* lfs, float* rhs, int size) {
  for (int i = 0; i < size; ++i) {
    if (lfs[i] != rhs[i]) {
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
  init_with_val(C_h, size_C, 0.0f);

  float* B_d_array[64];
  for (int i = 0; i < 64; ++i) {
    B_d_array[i] = B_d + i * 1024 * 512;
  }

  CUDA_CHECK(
      cudaMemcpy(A_d, A_h, size_A * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(B_d, B_h, size_B * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads_per_block(16);
  dim3 blocks_per_grid(256);

  default_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
      A_d, B_d, C_d_grand);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  expert_batch_matmul<<<blocks_per_grid, threads_per_block, 0, stream>>>(
      A_d, B_d_array, C_d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(C_h_grand, C_d_grand, size_C * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(C_h, C_d, size_C * sizeof(float), cudaMemcpyDeviceToHost));
  check_results(C_h_grand, C_h, size_C);

  return 0;
}
