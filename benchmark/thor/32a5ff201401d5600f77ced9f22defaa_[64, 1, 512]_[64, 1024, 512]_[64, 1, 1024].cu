
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(16) default_function_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[1];
  __shared__ float placeholder_d_shared[64];
  __shared__ float placeholder_shared[1024];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    ((float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 512) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 32))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 512) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 32))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder1 + ((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 64))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 512))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 1024))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 192))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 1536))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 320))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 2560))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 3072))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 448))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 3584))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 576))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 4608))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 5120))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 704))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 5632))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 832))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 6656))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 7168))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 960))))[0] = ((float4*)(placeholder1 + (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 4)) + 7680))))[0];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[((k_outer_inner * 16))] * placeholder_shared[(((((int)threadIdx.x) * 64) + (k_outer_inner * 16)))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 1))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 1))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 2))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 2))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 3))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 3))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 4))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 4))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 5))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 5))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 6))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 6))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 7))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 7))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 8))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 8))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 9))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 9))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 10))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 10))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 11))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 11))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 12))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 12))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 13))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 13))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 14))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 14))]));
      T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((k_outer_inner * 16) + 15))] * placeholder_shared[((((((int)threadIdx.x) * 64) + (k_outer_inner * 16)) + 15))]));
    }
  }
  T_batch_matmul_NT[(((((int)blockIdx.x) * 16) + ((int)threadIdx.x)))] = T_batch_matmul_NT_local[(0)];
}

