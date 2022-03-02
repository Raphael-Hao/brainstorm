#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long

extern "C" __global__ void __launch_bounds__(32) default_function_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT) {
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

