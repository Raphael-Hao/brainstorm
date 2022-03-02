#include <cuda_runtime.h>

__global__ void __launch_bounds__(128)
    gather_kernel_kernel_2_16_32_4(int* __restrict__ indices,
                                   float* __restrict__ T_gather,
                                   float* __restrict__ input) {
  __shared__ int indices_shared[1024];
  for (int ax0_inner = 0; ax0_inner < 8; ++ax0_inner) {
    indices_shared[((((((int)threadIdx.y) * 256) + (ax0_inner * 32)) +
                     ((int)threadIdx.x)))] =
        indices[(
            (((((((int)blockIdx.x) * 16384) + (((int)threadIdx.y) * 4096)) +
               (ax0_inner * 512)) +
              (((int)blockIdx.y) * 32)) +
             ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int ax0_inner_inner = 0; ax0_inner_inner < 8; ++ax0_inner_inner) {
    T_gather[((((((((int)blockIdx.x) * 16384) + (((int)threadIdx.y) * 4096)) +
                 (ax0_inner_inner * 512)) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x)))] =
        input[((((indices_shared[(
                      (((((int)threadIdx.y) * 256) + (ax0_inner_inner * 32)) +
                       ((int)threadIdx.x)))] *
                  512) +
                 (((int)blockIdx.y) * 32)) +
                ((int)threadIdx.x)))];
  }
}

__global__ void __launch_bounds__(128)
    gather_kernel_kernel_4_16_32_4(int* __restrict__ indices,
                                   float* __restrict__ T_gather,
                                   float* __restrict__ input) {
  __shared__ int indices_shared[512];
  for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
    indices_shared[((((((int)threadIdx.y) * 128) + (ax0_inner * 32)) +
                     ((int)threadIdx.x)))] =
        indices[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) +
                    (ax0_inner * 512)) +
                   (((int)blockIdx.y) * 32)) +
                  ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int ax0_inner_inner = 0; ax0_inner_inner < 4; ++ax0_inner_inner) {
    T_gather[((((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) +
                 (ax0_inner_inner * 512)) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x)))] =
        input[((((indices_shared[(
                      (((((int)threadIdx.y) * 128) + (ax0_inner_inner * 32)) +
                       ((int)threadIdx.x)))] *
                  512) +
                 (((int)blockIdx.y) * 32)) +
                ((int)threadIdx.x)))];
  }
}

__global__ void __launch_bounds__(128)
    gather_kernel_kernel_8_16_32_4(int* __restrict__ indices,
                                   float* __restrict__ T_gather,
                                   float* __restrict__ input) {
  __shared__ int indices_shared[256];
  for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
    indices_shared[((((((int)threadIdx.y) * 64) + (ax0_inner * 32)) +
                     ((int)threadIdx.x)))] =
        indices[((((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 1024)) +
                    (ax0_inner * 512)) +
                   (((int)blockIdx.y) * 32)) +
                  ((int)threadIdx.x)))];
  }
  __syncthreads();
  for (int ax0_inner_inner = 0; ax0_inner_inner < 2; ++ax0_inner_inner) {
    T_gather[((((((((int)blockIdx.x) * 4096) + (((int)threadIdx.y) * 1024)) +
                 (ax0_inner_inner * 512)) +
                (((int)blockIdx.y) * 32)) +
               ((int)threadIdx.x)))] =
        input[((((indices_shared[(
                      (((((int)threadIdx.y) * 64) + (ax0_inner_inner * 32)) +
                       ((int)threadIdx.x)))] *
                  512) +
                 (((int)blockIdx.y) * 32)) +
                ((int)threadIdx.x)))];
  }
}

__global__ void __launch_bounds__(128)
    gather_kernel_kernel_16_16_32_4(int* __restrict__ indices,
                                    float* __restrict__ T_gather,
                                    float* __restrict__ input) {
  __shared__ int indices_shared[128];
  indices_shared[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] =
      indices[(((((((int)blockIdx.x) * 2048) + (((int)threadIdx.y) * 512)) +
                 (((int)blockIdx.y) * 32)) +
                ((int)threadIdx.x)))];
  T_gather[(((((((int)blockIdx.x) * 2048) + (((int)threadIdx.y) * 512)) +
              (((int)blockIdx.y) * 32)) +
             ((int)threadIdx.x)))] =
      input[((
          ((indices_shared[(((((int)threadIdx.y) * 32) + ((int)threadIdx.x)))] *
            512) +
           (((int)blockIdx.y) * 32)) +
          ((int)threadIdx.x)))];
}
