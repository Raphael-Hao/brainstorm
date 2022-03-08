#pragma once
#include <cuda_runtime.h>
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long

__device__ __forceinline__ void matmul_32_1024_512(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT){
  float T_batch_matmul_NT_local[4];
  __shared__ float placeholder_d_shared[1024];
  __shared__ float placeholder_shared[2048];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(1)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(2)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(3)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(placeholder_d_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder + (((((((int)blockIdx.x) >> 5) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 5120))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6144))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 7168))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder1 + ((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 5120))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 7168))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 8192))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1152))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 9216))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1280))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 10240))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1408))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 11264))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 12288))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1664))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 13312))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1792))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 14336))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1920))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 7) * 524288) + ((((int)blockIdx.x) & 31) * 16384)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 15360))))[0];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)))] * placeholder_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)))]));
      T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 128))] * placeholder_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 1))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 1))]));
      T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 129))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 1))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 2))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 2))]));
      T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 130))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 2))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 3))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 3))]));
      T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 131))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 3))]));
      T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 256))] * placeholder_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)))]));
      T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 384))] * placeholder_shared[((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)))]));
      T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 257))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 1))]));
      T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 385))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 1))]));
      T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 258))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 2))]));
      T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 386))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 2))]));
      T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 259))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 3))]));
      T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 4) * 512) + (k_outer_inner * 4)) + 387))] * placeholder_shared[(((((((int)threadIdx.x) & 15) * 128) + (k_outer_inner * 4)) + 3))]));
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    T_batch_matmul_NT[(((((((((int)blockIdx.x) >> 5) * 4096) + ((((int)threadIdx.x) >> 4) * 2048)) + (i_inner * 512)) + ((((int)blockIdx.x) & 31) * 16)) + (((int)threadIdx.x) & 15)))] = T_batch_matmul_NT_local[(i_inner)];
  }
}

__device__ __forceinline__ void matmul_16_1024_512(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT){
  float T_batch_matmul_NT_local[8];
  __shared__ float placeholder_d_shared[2048];
  __shared__ float placeholder_shared[2048];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(1)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(2)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(3)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(4)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(5)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(6)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(7)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(placeholder_d_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder + (((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 5120))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6144))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 7168))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 8192))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 1152))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 9216))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 1280))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 10240))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 1408))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 11264))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 12288))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 1664))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 13312))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 1792))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 14336))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 1920))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 15360))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder1 + ((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 5120))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 7168))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 8192))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1152))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 9216))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1280))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 10240))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1408))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 11264))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 12288))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1664))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 13312))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1792))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 14336))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1920))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 15360))))[0];
    __syncthreads();
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[(((((int)threadIdx.x) & 1) * 1024))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 2))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 3))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 4))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 5))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 6))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 7))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 128))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 129))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 130))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 131))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 132))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 133))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 134))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 135))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 256))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 257))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 258))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 259))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 260))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 261))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 262))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 263))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 384))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 385))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 386))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 387))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 388))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 389))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 390))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 391))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 512))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 513))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 514))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 515))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 516))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 517))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 518))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 519))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 640))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 641))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 642))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 643))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 644))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 645))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 646))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 647))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 768))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 769))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 770))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 771))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 772))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 773))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 774))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 775))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 896))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 897))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 898))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 899))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 900))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 901))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 902))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 903))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 8))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 9))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 10))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 11))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 12))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 13))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 14))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 15))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 136))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 137))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 138))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 139))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 140))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 141))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 142))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 143))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 264))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 265))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 266))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 267))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 268))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 269))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 270))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 271))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 392))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 393))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 394))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 395))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 396))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 397))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 398))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 399))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 520))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 521))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 522))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 523))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 524))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 525))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 526))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 527))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 648))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 649))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 650))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 651))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 652))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 653))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 654))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 655))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 776))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 777))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 778))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 779))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 780))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 781))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 782))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 783))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 904))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 905))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 906))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 907))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 908))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 909))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 910))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 911))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 16))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 17))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 18))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 19))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 20))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 21))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 22))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 23))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 144))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 145))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 146))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 147))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 148))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 149))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 150))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 151))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 272))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 273))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 274))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 275))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 276))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 277))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 278))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 279))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 400))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 401))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 402))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 403))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 404))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 405))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 406))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 407))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 528))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 529))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 530))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 531))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 532))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 533))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 534))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 535))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 656))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 657))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 658))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 659))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 660))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 661))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 662))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 663))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 784))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 785))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 786))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 787))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 788))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 789))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 790))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 791))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 912))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 913))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 914))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 915))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 916))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 917))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 918))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 919))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 24))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 25))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 26))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 27))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 28))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 29))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 30))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 31))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 152))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 153))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 154))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 155))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 156))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 157))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 158))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 159))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 280))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 281))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 282))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 283))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 284))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 285))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 286))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 287))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 408))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 409))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 410))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 411))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 412))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 413))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 414))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 415))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 536))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 537))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 538))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 539))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 540))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 541))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 542))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 543))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 664))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 665))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 666))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 667))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 668))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 669))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 670))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 671))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 792))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 793))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 794))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 795))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 796))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 797))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 798))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 799))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 920))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 921))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 922))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 923))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 924))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 925))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 926))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 927))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 32))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 33))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 34))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 35))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 36))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 37))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 38))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 39))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 160))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 161))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 162))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 163))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 164))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 165))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 166))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 167))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 288))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 289))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 290))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 291))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 292))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 293))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 294))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 295))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 416))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 417))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 418))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 419))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 420))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 421))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 422))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 423))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 544))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 545))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 546))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 547))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 548))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 549))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 550))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 551))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 672))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 673))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 674))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 675))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 676))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 677))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 678))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 679))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 800))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 801))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 802))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 803))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 804))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 805))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 806))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 807))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 928))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 929))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 930))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 931))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 932))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 933))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 934))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 935))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 40))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 41))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 42))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 43))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 44))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 45))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 46))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 47))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 168))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 169))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 170))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 171))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 172))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 173))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 174))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 175))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 296))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 297))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 298))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 299))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 300))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 301))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 302))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 303))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 424))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 425))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 426))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 427))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 428))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 429))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 430))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 431))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 552))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 553))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 554))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 555))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 556))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 557))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 558))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 559))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 680))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 681))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 682))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 683))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 684))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 685))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 686))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 687))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 808))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 809))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 810))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 811))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 812))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 813))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 814))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 815))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 936))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 937))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 938))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 939))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 940))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 941))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 942))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 943))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 48))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 49))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 50))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 51))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 52))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 53))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 54))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 55))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 176))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 177))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 178))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 179))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 180))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 181))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 182))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 183))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 304))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 305))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 306))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 307))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 308))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 309))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 310))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 311))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 432))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 433))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 434))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 435))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 436))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 437))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 438))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 439))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 560))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 561))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 562))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 563))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 564))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 565))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 566))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 567))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 688))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 689))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 690))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 691))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 692))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 693))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 694))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 695))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 816))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 817))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 818))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 819))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 820))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 821))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 822))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 823))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 944))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 945))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 946))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 947))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 948))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 949))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 950))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 951))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 56))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 57))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 58))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 59))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 60))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 61))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 62))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 63))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 184))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 185))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 186))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 187))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 188))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 189))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 190))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 191))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 312))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 313))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 314))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 315))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 316))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 317))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 318))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 319))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 440))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 441))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 442))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 443))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 444))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 445))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 446))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 447))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 568))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 569))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 570))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 571))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 572))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 573))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 574))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 575))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 696))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 697))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 698))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 699))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 700))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 701))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 702))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 703))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 824))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 825))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 826))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 827))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 828))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 829))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 830))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 831))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 952))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 953))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 954))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 955))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 956))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 957))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 958))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 959))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 64))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 65))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 66))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 67))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 68))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 69))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 70))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 71))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 192))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 193))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 194))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 195))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 196))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 197))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 198))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 199))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 320))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 321))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 322))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 323))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 324))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 325))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 326))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 327))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 448))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 449))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 450))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 451))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 452))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 453))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 454))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 455))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 576))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 577))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 578))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 579))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 580))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 581))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 582))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 583))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 704))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 705))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 706))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 707))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 708))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 709))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 710))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 711))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 832))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 833))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 834))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 835))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 836))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 837))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 838))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 839))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 960))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 961))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 962))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 963))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 964))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 965))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 966))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 967))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 72))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 73))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 74))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 75))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 76))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 77))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 78))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 79))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 200))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 201))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 202))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 203))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 204))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 205))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 206))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 207))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 328))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 329))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 330))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 331))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 332))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 333))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 334))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 335))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 456))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 457))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 458))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 459))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 460))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 461))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 462))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 463))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 584))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 585))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 586))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 587))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 588))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 589))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 590))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 591))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 712))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 713))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 714))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 715))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 716))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 717))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 718))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 719))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 840))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 841))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 842))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 843))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 844))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 845))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 846))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 847))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 968))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 969))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 970))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 971))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 972))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 973))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 974))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 975))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 80))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 81))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 82))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 83))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 84))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 85))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 86))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 87))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 208))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 209))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 210))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 211))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 212))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 213))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 214))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 215))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 336))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 337))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 338))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 339))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 340))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 341))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 342))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 343))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 464))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 465))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 466))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 467))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 468))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 469))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 470))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 471))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 592))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 593))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 594))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 595))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 596))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 597))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 598))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 599))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 720))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 721))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 722))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 723))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 724))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 725))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 726))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 727))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 848))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 849))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 850))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 851))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 852))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 853))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 854))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 855))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 976))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 977))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 978))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 979))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 980))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 981))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 982))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 983))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 88))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 89))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 90))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 91))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 92))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 93))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 94))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 95))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 216))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 217))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 218))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 219))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 220))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 221))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 222))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 223))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 344))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 345))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 346))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 347))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 348))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 349))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 350))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 351))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 472))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 473))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 474))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 475))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 476))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 477))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 478))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 479))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 600))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 601))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 602))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 603))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 604))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 605))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 606))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 607))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 728))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 729))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 730))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 731))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 732))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 733))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 734))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 735))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 856))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 857))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 858))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 859))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 860))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 861))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 862))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 863))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 984))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 985))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 986))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 987))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 988))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 989))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 990))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 991))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 96))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 97))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 98))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 99))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 100))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 101))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 102))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 103))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 224))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 225))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 226))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 227))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 228))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 229))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 230))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 231))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 352))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 353))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 354))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 355))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 356))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 357))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 358))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 359))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 480))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 481))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 482))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 483))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 484))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 485))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 486))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 487))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 608))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 609))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 610))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 611))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 612))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 613))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 614))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 615))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 736))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 737))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 738))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 739))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 740))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 741))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 742))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 743))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 864))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 865))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 866))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 867))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 868))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 869))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 870))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 871))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 992))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 993))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 994))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 995))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 996))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 997))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 998))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 999))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 104))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 105))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 106))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 107))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 108))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 109))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 110))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 111))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 232))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 233))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 234))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 235))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 236))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 237))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 238))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 239))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 360))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 361))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 362))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 363))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 364))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 365))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 366))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 367))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 488))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 489))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 490))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 491))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 492))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 493))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 494))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 495))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 616))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 617))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 618))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 619))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 620))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 621))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 622))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 623))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 744))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 745))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 746))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 747))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 748))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 749))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 750))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 751))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 872))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 873))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 874))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 875))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 876))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 877))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 878))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 879))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1000))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1001))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1002))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1003))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1004))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1005))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1006))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1007))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 112))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 113))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 114))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 115))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 116))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 117))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 118))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 119))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 240))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 241))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 242))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 243))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 244))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 245))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 246))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 247))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 368))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 369))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 370))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 371))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 372))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 373))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 374))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 375))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 496))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 497))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 498))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 499))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 500))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 501))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 502))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 503))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 624))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 625))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 626))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 627))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 628))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 629))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 630))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 631))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 752))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 753))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 754))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 755))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 756))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 757))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 758))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 759))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 880))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 881))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 882))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 883))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 884))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 885))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 886))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 887))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1008))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1009))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1010))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1011))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1012))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1013))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1014))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1015))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 120))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 121))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 122))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 123))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 124))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 125))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 126))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 127))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 248))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 249))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 250))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 251))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 252))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 253))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 254))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 255))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 376))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 377))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 378))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 379))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 380))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 381))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 382))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 383))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 504))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 505))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 506))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 507))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 508))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 509))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 510))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 511))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 632))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 633))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 634))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 635))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 636))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 637))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 638))]));
    T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 639))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 760))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 761))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 762))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 763))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 764))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 765))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 766))]));
    T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 767))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 888))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 889))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 890))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 891))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 892))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 893))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 894))]));
    T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 895))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1016))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1017))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1018))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1019))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1020))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1021))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1022))]));
    T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 1024) + 1023))]));
  }
  for (int j_inner = 0; j_inner < 8; ++j_inner) {
    T_batch_matmul_NT[(((((((((int)blockIdx.x) >> 5) * 8192) + ((((int)threadIdx.x) >> 1) * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + j_inner))] = T_batch_matmul_NT_local[(j_inner)];
  }
}

__device__ __forceinline__ void matmul_8_1024_512(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT){
  float T_batch_matmul_NT_local[4];
  __shared__ float placeholder_d_shared[1024];
  __shared__ float placeholder_shared[1024];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(1)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(2)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(3)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(placeholder_d_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder + (((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 64))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 64))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 192))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + ((((((int)threadIdx.x) * 4) + 192) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 320))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + ((((((int)threadIdx.x) * 4) + 320) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 448))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + ((((((int)threadIdx.x) * 4) + 448) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 576))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + ((((((int)threadIdx.x) * 4) + 576) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 5120))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 704))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + ((((((int)threadIdx.x) * 4) + 704) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6144))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 832))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + ((((((int)threadIdx.x) * 4) + 832) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 7168))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 960))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 8192) + ((((((int)threadIdx.x) * 4) + 960) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder1 + ((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 64))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 64))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 192))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 192) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 320))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 320) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 448))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 448) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 576))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 576) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 5120))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 704))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 704) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 832))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 832) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 7168))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 960))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + ((((((int)threadIdx.x) * 4) + 960) >> 7) * 1024)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    __syncthreads();
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[(((((int)threadIdx.x) & 1) * 512))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 1))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 2))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 3))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 4))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 5))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 6))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 7))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 8))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 9))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 10))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 11))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 12))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 13))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 14))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 15))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 128))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 129))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 130))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 131))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 132))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 133))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 134))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 135))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 136))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 137))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 138))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 139))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 140))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 141))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 142))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 143))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 256))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 257))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 258))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 259))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 260))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 261))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 262))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 263))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 264))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 265))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 266))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 267))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 268))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 269))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 270))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 271))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(((((int)threadIdx.x) >> 1) * 128))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 384))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 1))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 385))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 2))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 386))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 3))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 387))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 4))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 388))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 5))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 389))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 6))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 390))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 7))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 391))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 8))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 392))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 9))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 393))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 10))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 394))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 11))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 395))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 12))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 396))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 13))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 397))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 14))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 398))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 15))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 399))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 16))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 17))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 18))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 19))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 20))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 21))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 22))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 23))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 24))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 25))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 26))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 27))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 28))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 29))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 30))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 31))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 144))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 145))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 146))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 147))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 148))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 149))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 150))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 151))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 152))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 153))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 154))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 155))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 156))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 157))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 158))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 159))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 272))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 273))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 274))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 275))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 276))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 277))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 278))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 279))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 280))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 281))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 282))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 283))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 284))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 285))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 286))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 287))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 16))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 400))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 17))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 401))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 18))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 402))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 19))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 403))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 20))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 404))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 21))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 405))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 22))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 406))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 23))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 407))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 24))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 408))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 25))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 409))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 26))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 410))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 27))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 411))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 28))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 412))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 29))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 413))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 30))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 414))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 31))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 415))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 32))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 33))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 34))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 35))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 36))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 37))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 38))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 39))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 40))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 41))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 42))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 43))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 44))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 45))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 46))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 47))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 160))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 161))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 162))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 163))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 164))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 165))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 166))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 167))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 168))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 169))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 170))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 171))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 172))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 173))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 174))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 175))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 288))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 289))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 290))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 291))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 292))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 293))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 294))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 295))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 296))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 297))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 298))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 299))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 300))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 301))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 302))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 303))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 32))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 416))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 33))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 417))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 34))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 418))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 35))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 419))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 36))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 420))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 37))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 421))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 38))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 422))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 39))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 423))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 40))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 424))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 41))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 425))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 42))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 426))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 43))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 427))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 44))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 428))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 45))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 429))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 46))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 430))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 47))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 431))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 48))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 49))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 50))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 51))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 52))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 53))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 54))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 55))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 56))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 57))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 58))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 59))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 60))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 61))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 62))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 63))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 176))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 177))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 178))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 179))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 180))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 181))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 182))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 183))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 184))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 185))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 186))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 187))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 188))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 189))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 190))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 191))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 304))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 305))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 306))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 307))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 308))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 309))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 310))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 311))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 312))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 313))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 314))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 315))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 316))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 317))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 318))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 319))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 48))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 432))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 49))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 433))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 50))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 434))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 51))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 435))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 52))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 436))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 53))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 437))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 54))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 438))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 55))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 439))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 56))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 440))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 57))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 441))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 58))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 442))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 59))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 443))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 60))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 444))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 61))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 445))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 62))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 446))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 63))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 447))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 64))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 65))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 66))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 67))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 68))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 69))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 70))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 71))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 72))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 73))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 74))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 75))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 76))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 77))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 78))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 79))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 192))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 193))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 194))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 195))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 196))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 197))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 198))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 199))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 200))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 201))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 202))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 203))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 204))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 205))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 206))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 207))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 320))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 321))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 322))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 323))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 324))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 325))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 326))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 327))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 328))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 329))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 330))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 331))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 332))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 333))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 334))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 335))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 64))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 448))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 65))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 449))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 66))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 450))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 67))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 451))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 68))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 452))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 69))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 453))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 70))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 454))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 71))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 455))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 72))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 456))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 73))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 457))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 74))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 458))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 75))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 459))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 76))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 460))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 77))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 461))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 78))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 462))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 79))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 463))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 80))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 81))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 82))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 83))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 84))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 85))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 86))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 87))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 88))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 89))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 90))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 91))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 92))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 93))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 94))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 95))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 208))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 209))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 210))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 211))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 212))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 213))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 214))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 215))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 216))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 217))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 218))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 219))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 220))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 221))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 222))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 223))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 336))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 337))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 338))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 339))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 340))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 341))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 342))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 343))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 344))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 345))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 346))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 347))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 348))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 349))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 350))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 351))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 80))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 464))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 81))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 465))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 82))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 466))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 83))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 467))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 84))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 468))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 85))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 469))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 86))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 470))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 87))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 471))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 88))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 472))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 89))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 473))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 90))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 474))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 91))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 475))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 92))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 476))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 93))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 477))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 94))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 478))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 95))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 479))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 96))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 97))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 98))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 99))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 100))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 101))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 102))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 103))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 104))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 105))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 106))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 107))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 108))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 109))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 110))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 111))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 224))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 225))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 226))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 227))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 228))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 229))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 230))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 231))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 232))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 233))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 234))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 235))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 236))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 237))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 238))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 239))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 352))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 353))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 354))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 355))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 356))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 357))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 358))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 359))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 360))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 361))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 362))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 363))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 364))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 365))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 366))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 367))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 96))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 480))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 97))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 481))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 98))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 482))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 99))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 483))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 100))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 484))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 101))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 485))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 102))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 486))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 103))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 487))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 104))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 488))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 105))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 489))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 106))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 490))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 107))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 491))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 108))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 492))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 109))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 493))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 110))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 494))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 111))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 495))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 112))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 113))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 114))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 115))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 116))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 117))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 118))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 119))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 120))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 121))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 122))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 123))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 124))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 125))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 126))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 127))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 240))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 241))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 242))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 243))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 244))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 245))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 246))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 247))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 248))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 249))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 250))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 251))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 252))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 253))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 254))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 255))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 368))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 369))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 370))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 371))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 372))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 373))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 374))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 375))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 376))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 377))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 378))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 379))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 380))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 381))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 382))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 383))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 112))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 496))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 113))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 497))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 114))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 498))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 115))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 499))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 116))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 500))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 117))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 501))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 118))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 502))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 119))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 503))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 120))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 504))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 121))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 505))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 122))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 506))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 123))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 507))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 124))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 508))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 125))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 509))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 126))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 510))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[((((((int)threadIdx.x) >> 1) * 128) + 127))] * placeholder_shared[((((((int)threadIdx.x) & 1) * 512) + 511))]));
  }
  for (int j_inner = 0; j_inner < 4; ++j_inner) {
    T_batch_matmul_NT[(((((((((int)blockIdx.x) >> 6) * 4096) + ((((int)threadIdx.x) >> 1) * 512)) + ((((int)blockIdx.x) & 63) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + j_inner))] = T_batch_matmul_NT_local[(j_inner)];
  }
}

__device__ __forceinline__ void matmul_4_1024_512(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT){
  float T_batch_matmul_NT_local[4];
  __shared__ float placeholder_d_shared[1024];
  __shared__ float placeholder_shared[4096];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(2)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(1)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(3)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(placeholder_d_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder + (((((((int)blockIdx.x) >> 5) * 4096) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 64))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 64))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 128))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 192))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 192))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 320))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 320) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 384) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 448))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 448) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 576))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 576) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 640) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 704))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 704) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 832))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 832) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 896) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_d_shared + (((((int)threadIdx.x) * 4) + 960))))[0] = ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 5) * 4096) + ((((((int)threadIdx.x) * 4) + 960) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder1 + ((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 64))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 64))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 128))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 192))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 192))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 320))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 320) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 384) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 448))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 448) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 576))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 576) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 640) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 704))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 704) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 832))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 832) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 896) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 960))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 960) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1088))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1088) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1152))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1152) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1216))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1216) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1280))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 5120))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1344))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1344) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1408))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1408) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1472))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1472) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1600))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1600) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1664))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1664) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1728))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1728) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1792))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 7168))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1856))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1856) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1920))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1920) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1984))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 1984) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2048))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 8192))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2112))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2112) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2176))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2176) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2240))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2240) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2304))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 9216))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2368))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2368) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2432))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2432) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2496))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2496) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2560))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 10240))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2624))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2624) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2688))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2688) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2752))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2752) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2816))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 11264))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2880))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2880) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2944))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 2944) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3008))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3008) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3072))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 12288))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3136))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3136) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3200))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3200) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3264))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3264) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3328))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 13312))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3392))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3392) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3456))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3456) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3520))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3520) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3584))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 14336))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3648))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3648) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3712))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3712) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3776))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3776) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3840))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + (k_outer_outer * 256)) + (((int)threadIdx.x) * 4)) + 15360))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3904))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3904) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3968))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 3968) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 4032))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 16384) + ((((((int)threadIdx.x) * 4) + 4032) >> 8) * 1024)) + (k_outer_outer * 256)) + ((((int)threadIdx.x) * 4) + 192)))))[0];
    __syncthreads();
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(0)] * placeholder_shared[((((int)threadIdx.x) * 256))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(512)] * placeholder_shared[((((int)threadIdx.x) * 256))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(1)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 1))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(513)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 1))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(2)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 2))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(514)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 2))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(3)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 3))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(515)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 3))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(4)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 4))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(516)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 4))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(5)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 5))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(517)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 5))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(6)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 6))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(518)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 6))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(7)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 7))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(519)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 7))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(8)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 8))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(520)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 8))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(9)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 9))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(521)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 9))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(10)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 10))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(522)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 10))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(11)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 11))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(523)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 11))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(12)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 12))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(524)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 12))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(13)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 13))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(525)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 13))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(14)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 14))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(526)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 14))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(15)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 15))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(527)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 15))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(256)] * placeholder_shared[((((int)threadIdx.x) * 256))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(768)] * placeholder_shared[((((int)threadIdx.x) * 256))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(257)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 1))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(769)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 1))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(258)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 2))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(770)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 2))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(259)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 3))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(771)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 3))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(260)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 4))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(772)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 4))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(261)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 5))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(773)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 5))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(262)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 6))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(774)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 6))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(263)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 7))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(775)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 7))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(264)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 8))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(776)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 8))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(265)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 9))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(777)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 9))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(266)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 10))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(778)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 10))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(267)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 11))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(779)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 11))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(268)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 12))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(780)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 12))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(269)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 13))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(781)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 13))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(270)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 14))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(782)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 14))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(271)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 15))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(783)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 15))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(16)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 16))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(528)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 16))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(17)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 17))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(529)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 17))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(18)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 18))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(530)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 18))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(19)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 19))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(531)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 19))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(20)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 20))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(532)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 20))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(21)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 21))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(533)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 21))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(22)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 22))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(534)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 22))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(23)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 23))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(535)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 23))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(24)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 24))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(536)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 24))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(25)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 25))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(537)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 25))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(26)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 26))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(538)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 26))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(27)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 27))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(539)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 27))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(28)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 28))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(540)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 28))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(29)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 29))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(541)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 29))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(30)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 30))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(542)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 30))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(31)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 31))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(543)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 31))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(272)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 16))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(784)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 16))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(273)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 17))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(785)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 17))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(274)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 18))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(786)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 18))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(275)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 19))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(787)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 19))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(276)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 20))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(788)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 20))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(277)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 21))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(789)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 21))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(278)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 22))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(790)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 22))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(279)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 23))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(791)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 23))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(280)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 24))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(792)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 24))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(281)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 25))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(793)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 25))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(282)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 26))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(794)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 26))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(283)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 27))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(795)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 27))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(284)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 28))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(796)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 28))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(285)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 29))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(797)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 29))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(286)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 30))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(798)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 30))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(287)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 31))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(799)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 31))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(32)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 32))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(544)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 32))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(33)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 33))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(545)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 33))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(34)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 34))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(546)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 34))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(35)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 35))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(547)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 35))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(36)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 36))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(548)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 36))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(37)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 37))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(549)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 37))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(38)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 38))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(550)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 38))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(39)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 39))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(551)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 39))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(40)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 40))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(552)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 40))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(41)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 41))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(553)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 41))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(42)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 42))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(554)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 42))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(43)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 43))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(555)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 43))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(44)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 44))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(556)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 44))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(45)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 45))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(557)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 45))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(46)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 46))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(558)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 46))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(47)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 47))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(559)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 47))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(288)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 32))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(800)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 32))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(289)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 33))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(801)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 33))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(290)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 34))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(802)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 34))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(291)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 35))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(803)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 35))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(292)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 36))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(804)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 36))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(293)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 37))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(805)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 37))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(294)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 38))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(806)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 38))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(295)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 39))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(807)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 39))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(296)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 40))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(808)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 40))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(297)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 41))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(809)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 41))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(298)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 42))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(810)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 42))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(299)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 43))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(811)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 43))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(300)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 44))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(812)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 44))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(301)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 45))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(813)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 45))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(302)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 46))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(814)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 46))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(303)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 47))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(815)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 47))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(48)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 48))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(560)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 48))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(49)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 49))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(561)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 49))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(50)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 50))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(562)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 50))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(51)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 51))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(563)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 51))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(52)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 52))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(564)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 52))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(53)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 53))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(565)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 53))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(54)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 54))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(566)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 54))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(55)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 55))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(567)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 55))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(56)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 56))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(568)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 56))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(57)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 57))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(569)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 57))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(58)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 58))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(570)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 58))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(59)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 59))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(571)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 59))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(60)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 60))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(572)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 60))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(61)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 61))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(573)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 61))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(62)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 62))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(574)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 62))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(63)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 63))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(575)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 63))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(304)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 48))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(816)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 48))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(305)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 49))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(817)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 49))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(306)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 50))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(818)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 50))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(307)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 51))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(819)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 51))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(308)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 52))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(820)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 52))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(309)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 53))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(821)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 53))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(310)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 54))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(822)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 54))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(311)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 55))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(823)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 55))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(312)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 56))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(824)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 56))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(313)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 57))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(825)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 57))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(314)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 58))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(826)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 58))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(315)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 59))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(827)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 59))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(316)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 60))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(828)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 60))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(317)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 61))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(829)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 61))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(318)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 62))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(830)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 62))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(319)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 63))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(831)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 63))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(64)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 64))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(576)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 64))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(65)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 65))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(577)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 65))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(66)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 66))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(578)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 66))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(67)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 67))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(579)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 67))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(68)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 68))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(580)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 68))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(69)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 69))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(581)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 69))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(70)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 70))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(582)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 70))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(71)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 71))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(583)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 71))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(72)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 72))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(584)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 72))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(73)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 73))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(585)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 73))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(74)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 74))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(586)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 74))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(75)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 75))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(587)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 75))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(76)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 76))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(588)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 76))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(77)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 77))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(589)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 77))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(78)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 78))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(590)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 78))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(79)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 79))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(591)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 79))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(320)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 64))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(832)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 64))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(321)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 65))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(833)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 65))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(322)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 66))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(834)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 66))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(323)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 67))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(835)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 67))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(324)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 68))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(836)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 68))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(325)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 69))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(837)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 69))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(326)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 70))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(838)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 70))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(327)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 71))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(839)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 71))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(328)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 72))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(840)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 72))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(329)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 73))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(841)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 73))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(330)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 74))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(842)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 74))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(331)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 75))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(843)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 75))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(332)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 76))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(844)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 76))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(333)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 77))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(845)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 77))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(334)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 78))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(846)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 78))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(335)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 79))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(847)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 79))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(80)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 80))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(592)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 80))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(81)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 81))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(593)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 81))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(82)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 82))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(594)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 82))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(83)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 83))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(595)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 83))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(84)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 84))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(596)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 84))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(85)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 85))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(597)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 85))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(86)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 86))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(598)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 86))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(87)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 87))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(599)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 87))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(88)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 88))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(600)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 88))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(89)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 89))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(601)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 89))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(90)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 90))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(602)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 90))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(91)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 91))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(603)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 91))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(92)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 92))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(604)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 92))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(93)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 93))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(605)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 93))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(94)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 94))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(606)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 94))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(95)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 95))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(607)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 95))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(336)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 80))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(848)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 80))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(337)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 81))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(849)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 81))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(338)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 82))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(850)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 82))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(339)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 83))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(851)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 83))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(340)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 84))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(852)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 84))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(341)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 85))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(853)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 85))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(342)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 86))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(854)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 86))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(343)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 87))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(855)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 87))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(344)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 88))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(856)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 88))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(345)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 89))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(857)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 89))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(346)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 90))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(858)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 90))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(347)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 91))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(859)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 91))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(348)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 92))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(860)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 92))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(349)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 93))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(861)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 93))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(350)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 94))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(862)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 94))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(351)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 95))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(863)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 95))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(96)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 96))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(608)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 96))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(97)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 97))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(609)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 97))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(98)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 98))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(610)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 98))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(99)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 99))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(611)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 99))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(100)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 100))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(612)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 100))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(101)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 101))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(613)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 101))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(102)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 102))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(614)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 102))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(103)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 103))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(615)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 103))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(104)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 104))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(616)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 104))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(105)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 105))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(617)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 105))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(106)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 106))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(618)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 106))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(107)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 107))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(619)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 107))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(108)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 108))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(620)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 108))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(109)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 109))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(621)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 109))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(110)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 110))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(622)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 110))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(111)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 111))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(623)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 111))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(352)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 96))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(864)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 96))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(353)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 97))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(865)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 97))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(354)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 98))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(866)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 98))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(355)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 99))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(867)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 99))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(356)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 100))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(868)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 100))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(357)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 101))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(869)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 101))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(358)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 102))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(870)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 102))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(359)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 103))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(871)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 103))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(360)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 104))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(872)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 104))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(361)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 105))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(873)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 105))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(362)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 106))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(874)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 106))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(363)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 107))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(875)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 107))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(364)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 108))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(876)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 108))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(365)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 109))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(877)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 109))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(366)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 110))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(878)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 110))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(367)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 111))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(879)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 111))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(112)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 112))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(624)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 112))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(113)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 113))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(625)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 113))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(114)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 114))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(626)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 114))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(115)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 115))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(627)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 115))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(116)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 116))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(628)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 116))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(117)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 117))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(629)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 117))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(118)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 118))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(630)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 118))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(119)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 119))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(631)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 119))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(120)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 120))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(632)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 120))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(121)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 121))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(633)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 121))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(122)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 122))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(634)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 122))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(123)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 123))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(635)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 123))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(124)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 124))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(636)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 124))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(125)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 125))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(637)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 125))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(126)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 126))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(638)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 126))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(127)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 127))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(639)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 127))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(368)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 112))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(880)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 112))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(369)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 113))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(881)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 113))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(370)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 114))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(882)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 114))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(371)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 115))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(883)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 115))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(372)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 116))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(884)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 116))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(373)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 117))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(885)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 117))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(374)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 118))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(886)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 118))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(375)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 119))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(887)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 119))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(376)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 120))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(888)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 120))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(377)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 121))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(889)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 121))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(378)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 122))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(890)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 122))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(379)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 123))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(891)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 123))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(380)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 124))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(892)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 124))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(381)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 125))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(893)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 125))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(382)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 126))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(894)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 126))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(383)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 127))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(895)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 127))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(128)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 128))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(640)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 128))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(129)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 129))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(641)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 129))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(130)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 130))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(642)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 130))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(131)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 131))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(643)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 131))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(132)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 132))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(644)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 132))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(133)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 133))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(645)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 133))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(134)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 134))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(646)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 134))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(135)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 135))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(647)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 135))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(136)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 136))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(648)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 136))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(137)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 137))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(649)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 137))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(138)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 138))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(650)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 138))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(139)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 139))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(651)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 139))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(140)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 140))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(652)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 140))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(141)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 141))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(653)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 141))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(142)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 142))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(654)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 142))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(143)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 143))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(655)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 143))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(384)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 128))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(896)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 128))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(385)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 129))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(897)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 129))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(386)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 130))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(898)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 130))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(387)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 131))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(899)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 131))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(388)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 132))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(900)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 132))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(389)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 133))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(901)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 133))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(390)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 134))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(902)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 134))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(391)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 135))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(903)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 135))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(392)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 136))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(904)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 136))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(393)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 137))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(905)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 137))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(394)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 138))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(906)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 138))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(395)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 139))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(907)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 139))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(396)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 140))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(908)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 140))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(397)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 141))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(909)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 141))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(398)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 142))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(910)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 142))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(399)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 143))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(911)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 143))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(144)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 144))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(656)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 144))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(145)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 145))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(657)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 145))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(146)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 146))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(658)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 146))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(147)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 147))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(659)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 147))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(148)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 148))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(660)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 148))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(149)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 149))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(661)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 149))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(150)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 150))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(662)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 150))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(151)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 151))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(663)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 151))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(152)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 152))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(664)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 152))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(153)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 153))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(665)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 153))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(154)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 154))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(666)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 154))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(155)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 155))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(667)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 155))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(156)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 156))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(668)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 156))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(157)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 157))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(669)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 157))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(158)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 158))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(670)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 158))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(159)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 159))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(671)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 159))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(400)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 144))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(912)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 144))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(401)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 145))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(913)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 145))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(402)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 146))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(914)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 146))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(403)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 147))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(915)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 147))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(404)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 148))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(916)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 148))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(405)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 149))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(917)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 149))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(406)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 150))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(918)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 150))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(407)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 151))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(919)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 151))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(408)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 152))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(920)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 152))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(409)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 153))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(921)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 153))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(410)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 154))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(922)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 154))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(411)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 155))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(923)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 155))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(412)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 156))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(924)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 156))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(413)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 157))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(925)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 157))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(414)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 158))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(926)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 158))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(415)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 159))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(927)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 159))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(160)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 160))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(672)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 160))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(161)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 161))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(673)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 161))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(162)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 162))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(674)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 162))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(163)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 163))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(675)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 163))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(164)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 164))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(676)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 164))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(165)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 165))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(677)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 165))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(166)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 166))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(678)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 166))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(167)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 167))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(679)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 167))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(168)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 168))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(680)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 168))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(169)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 169))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(681)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 169))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(170)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 170))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(682)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 170))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(171)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 171))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(683)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 171))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(172)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 172))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(684)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 172))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(173)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 173))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(685)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 173))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(174)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 174))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(686)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 174))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(175)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 175))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(687)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 175))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(416)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 160))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(928)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 160))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(417)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 161))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(929)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 161))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(418)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 162))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(930)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 162))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(419)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 163))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(931)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 163))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(420)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 164))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(932)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 164))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(421)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 165))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(933)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 165))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(422)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 166))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(934)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 166))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(423)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 167))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(935)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 167))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(424)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 168))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(936)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 168))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(425)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 169))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(937)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 169))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(426)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 170))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(938)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 170))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(427)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 171))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(939)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 171))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(428)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 172))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(940)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 172))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(429)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 173))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(941)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 173))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(430)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 174))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(942)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 174))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(431)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 175))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(943)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 175))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(176)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 176))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(688)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 176))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(177)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 177))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(689)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 177))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(178)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 178))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(690)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 178))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(179)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 179))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(691)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 179))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(180)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 180))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(692)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 180))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(181)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 181))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(693)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 181))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(182)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 182))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(694)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 182))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(183)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 183))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(695)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 183))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(184)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 184))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(696)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 184))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(185)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 185))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(697)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 185))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(186)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 186))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(698)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 186))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(187)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 187))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(699)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 187))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(188)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 188))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(700)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 188))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(189)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 189))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(701)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 189))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(190)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 190))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(702)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 190))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(191)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 191))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(703)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 191))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(432)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 176))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(944)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 176))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(433)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 177))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(945)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 177))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(434)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 178))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(946)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 178))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(435)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 179))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(947)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 179))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(436)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 180))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(948)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 180))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(437)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 181))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(949)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 181))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(438)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 182))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(950)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 182))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(439)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 183))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(951)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 183))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(440)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 184))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(952)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 184))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(441)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 185))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(953)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 185))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(442)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 186))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(954)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 186))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(443)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 187))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(955)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 187))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(444)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 188))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(956)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 188))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(445)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 189))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(957)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 189))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(446)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 190))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(958)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 190))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(447)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 191))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(959)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 191))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(192)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 192))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(704)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 192))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(193)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 193))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(705)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 193))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(194)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 194))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(706)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 194))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(195)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 195))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(707)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 195))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(196)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 196))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(708)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 196))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(197)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 197))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(709)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 197))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(198)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 198))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(710)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 198))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(199)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 199))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(711)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 199))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(200)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 200))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(712)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 200))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(201)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 201))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(713)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 201))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(202)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 202))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(714)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 202))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(203)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 203))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(715)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 203))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(204)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 204))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(716)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 204))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(205)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 205))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(717)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 205))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(206)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 206))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(718)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 206))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(207)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 207))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(719)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 207))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(448)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 192))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(960)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 192))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(449)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 193))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(961)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 193))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(450)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 194))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(962)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 194))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(451)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 195))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(963)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 195))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(452)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 196))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(964)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 196))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(453)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 197))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(965)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 197))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(454)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 198))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(966)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 198))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(455)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 199))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(967)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 199))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(456)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 200))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(968)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 200))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(457)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 201))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(969)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 201))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(458)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 202))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(970)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 202))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(459)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 203))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(971)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 203))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(460)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 204))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(972)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 204))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(461)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 205))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(973)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 205))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(462)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 206))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(974)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 206))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(463)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 207))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(975)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 207))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(208)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 208))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(720)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 208))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(209)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 209))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(721)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 209))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(210)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 210))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(722)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 210))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(211)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 211))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(723)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 211))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(212)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 212))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(724)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 212))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(213)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 213))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(725)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 213))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(214)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 214))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(726)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 214))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(215)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 215))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(727)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 215))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(216)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 216))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(728)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 216))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(217)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 217))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(729)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 217))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(218)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 218))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(730)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 218))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(219)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 219))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(731)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 219))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(220)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 220))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(732)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 220))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(221)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 221))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(733)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 221))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(222)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 222))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(734)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 222))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(223)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 223))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(735)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 223))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(464)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 208))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(976)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 208))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(465)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 209))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(977)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 209))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(466)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 210))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(978)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 210))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(467)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 211))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(979)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 211))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(468)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 212))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(980)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 212))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(469)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 213))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(981)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 213))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(470)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 214))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(982)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 214))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(471)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 215))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(983)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 215))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(472)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 216))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(984)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 216))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(473)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 217))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(985)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 217))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(474)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 218))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(986)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 218))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(475)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 219))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(987)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 219))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(476)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 220))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(988)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 220))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(477)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 221))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(989)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 221))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(478)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 222))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(990)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 222))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(479)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 223))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(991)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 223))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(224)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 224))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(736)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 224))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(225)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 225))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(737)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 225))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(226)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 226))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(738)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 226))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(227)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 227))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(739)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 227))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(228)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 228))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(740)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 228))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(229)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 229))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(741)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 229))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(230)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 230))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(742)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 230))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(231)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 231))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(743)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 231))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(232)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 232))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(744)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 232))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(233)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 233))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(745)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 233))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(234)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 234))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(746)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 234))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(235)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 235))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(747)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 235))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(236)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 236))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(748)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 236))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(237)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 237))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(749)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 237))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(238)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 238))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(750)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 238))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(239)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 239))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(751)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 239))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(480)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 224))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(992)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 224))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(481)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 225))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(993)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 225))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(482)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 226))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(994)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 226))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(483)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 227))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(995)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 227))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(484)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 228))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(996)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 228))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(485)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 229))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(997)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 229))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(486)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 230))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(998)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 230))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(487)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 231))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(999)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 231))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(488)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 232))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1000)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 232))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(489)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 233))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1001)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 233))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(490)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 234))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1002)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 234))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(491)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 235))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1003)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 235))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(492)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 236))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1004)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 236))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(493)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 237))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1005)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 237))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(494)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 238))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1006)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 238))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(495)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 239))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1007)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 239))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(240)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 240))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(752)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 240))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(241)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 241))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(753)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 241))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(242)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 242))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(754)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 242))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(243)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 243))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(755)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 243))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(244)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 244))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(756)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 244))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(245)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 245))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(757)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 245))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(246)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 246))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(758)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 246))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(247)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 247))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(759)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 247))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(248)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 248))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(760)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 248))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(249)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 249))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(761)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 249))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(250)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 250))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(762)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 250))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(251)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 251))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(763)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 251))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(252)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 252))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(764)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 252))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(253)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 253))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(765)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 253))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(254)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 254))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(766)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 254))]));
    T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(255)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 255))]));
    T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(767)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 255))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(496)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 240))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1008)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 240))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(497)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 241))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1009)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 241))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(498)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 242))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1010)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 242))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(499)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 243))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1011)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 243))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(500)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 244))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1012)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 244))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(501)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 245))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1013)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 245))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(502)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 246))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1014)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 246))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(503)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 247))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1015)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 247))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(504)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 248))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1016)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 248))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(505)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 249))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1017)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 249))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(506)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 250))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1018)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 250))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(507)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 251))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1019)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 251))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(508)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 252))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1020)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 252))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(509)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 253))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1021)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 253))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(510)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 254))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1022)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 254))]));
    T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(511)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 255))]));
    T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(1023)] * placeholder_shared[(((((int)threadIdx.x) * 256) + 255))]));
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    T_batch_matmul_NT[((((((((int)blockIdx.x) >> 5) * 2048) + (i_inner * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((int)threadIdx.x)))] = T_batch_matmul_NT_local[(i_inner)];
    T_batch_matmul_NT[(((((((((int)blockIdx.x) >> 5) * 2048) + (i_inner * 512)) + ((((int)blockIdx.x) & 31) * 16)) + ((int)threadIdx.x)) + 1024))] = T_batch_matmul_NT_local[((i_inner + 2))];
  }
}

__device__ __forceinline__ void matmul_2_1024_512(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT){
  float T_batch_matmul_NT_local[2];
  __shared__ float placeholder_d_shared[128];
  __shared__ float placeholder_shared[2048];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(1)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(placeholder_d_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)(placeholder + ((((((((int)blockIdx.x) >> 4) * 2048) +
                                     ((((int)threadIdx.x) >> 4) * 1024)) +
                                    (k_outer_outer * 64)) +
                                   ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 32768) +
                                      ((((int)threadIdx.x) >> 4) * 1024)) +
                                     (k_outer_outer * 64)) +
                                    ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    8192))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    10240))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    12288))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    14336))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    16384))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1152))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    18432))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1280))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    20480))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1408))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    22528))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    24576))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1664))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    26624))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1792))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    28672))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1920))))[0] =
        ((float4*)(placeholder1 + ((((((((int)blockIdx.x) * 32768) +
                                       ((((int)threadIdx.x) >> 4) * 1024)) +
                                      (k_outer_outer * 64)) +
                                     ((((int)threadIdx.x) & 15) * 4)) +
                                    30720))))[0];
    __syncthreads();
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(0)] *
          placeholder_shared[((((int)threadIdx.x) * 64))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(64)] *
          placeholder_shared[((((int)threadIdx.x) * 64))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(1)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 1))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(65)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 1))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(2)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 2))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(66)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 2))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(3)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 3))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(67)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 3))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(4)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 4))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(68)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 4))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(5)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 5))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(69)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 5))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(6)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 6))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(70)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 6))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(7)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 7))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(71)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 7))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(8)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 8))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(72)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 8))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(9)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 9))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(73)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 9))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(10)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 10))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(74)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 10))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(11)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 11))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(75)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 11))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(12)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 12))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(76)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 12))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(13)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 13))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(77)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 13))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(14)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 14))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(78)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 14))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(15)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 15))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(79)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 15))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(16)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 16))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(80)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 16))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(17)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 17))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(81)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 17))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(18)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 18))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(82)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 18))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(19)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 19))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(83)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 19))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(20)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 20))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(84)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 20))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(21)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 21))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(85)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 21))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(22)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 22))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(86)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 22))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(23)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 23))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(87)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 23))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(24)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 24))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(88)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 24))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(25)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 25))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(89)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 25))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(26)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 26))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(90)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 26))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(27)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 27))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(91)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 27))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(28)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 28))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(92)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 28))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(29)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 29))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(93)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 29))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(30)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 30))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(94)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 30))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(31)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 31))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(95)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 31))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(32)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 32))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(96)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 32))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(33)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 33))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(97)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 33))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(34)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 34))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(98)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 34))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(35)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 35))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(99)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 35))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(36)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 36))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(100)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 36))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(37)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 37))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(101)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 37))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(38)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 38))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(102)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 38))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(39)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 39))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(103)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 39))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(40)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 40))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(104)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 40))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(41)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 41))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(105)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 41))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(42)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 42))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(106)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 42))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(43)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 43))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(107)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 43))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(44)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 44))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(108)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 44))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(45)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 45))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(109)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 45))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(46)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 46))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(110)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 46))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(47)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 47))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(111)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 47))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(48)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 48))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(112)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 48))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(49)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 49))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(113)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 49))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(50)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 50))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(114)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 50))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(51)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 51))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(115)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 51))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(52)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 52))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(116)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 52))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(53)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 53))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(117)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 53))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(54)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 54))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(118)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 54))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(55)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 55))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(119)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 55))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(56)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 56))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(120)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 56))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(57)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 57))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(121)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 57))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(58)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 58))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(122)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 58))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(59)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 59))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(123)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 59))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(60)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 60))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(124)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 60))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(61)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 61))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(125)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 61))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(62)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 62))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(126)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 62))]));
    T_batch_matmul_NT_local[(0)] =
        (T_batch_matmul_NT_local[(0)] +
         (placeholder_d_shared[(63)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 63))]));
    T_batch_matmul_NT_local[(1)] =
        (T_batch_matmul_NT_local[(1)] +
         (placeholder_d_shared[(127)] *
          placeholder_shared[(((((int)threadIdx.x) * 64) + 63))]));
  }
  T_batch_matmul_NT[(
      ((((((int)blockIdx.x) >> 4) * 1024) + ((((int)blockIdx.x) & 15) * 32)) +
       ((int)threadIdx.x)))] = T_batch_matmul_NT_local[(0)];
  T_batch_matmul_NT[(
      (((((((int)blockIdx.x) >> 4) * 1024) + ((((int)blockIdx.x) & 15) * 32)) +
        ((int)threadIdx.x)) +
       512))] = T_batch_matmul_NT_local[(1)];
}

// calling 128 threads per block and using 256 blocks
__device__ __forceinline__ void matmul_1_1024_512(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[1];
  __shared__ float placeholder_d_shared[512];
  __shared__ float placeholder_shared[65536];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  // T_batch_matmul_NT_local[(1)] = 0.000000e+00f;
#pragma unroll
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    ((float4*)(placeholder_d_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)(placeholder + (k_outer_outer * 512)+ ((((((int)blockIdx.x) >> 1) * 4096) +
                                   (((int)threadIdx.x) * 4)))))[0];
    for (int k_)
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

__device__ __forceinline__ void matmul_weight_1024_512(float* __restrict__ placeholder, float* __restrict__ placeholder1[64], float* __restrict__ T_batch_matmul_NT){
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