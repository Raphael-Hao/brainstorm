
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
extern "C" __global__ void __launch_bounds__(32) default_function_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[16];
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
  T_batch_matmul_NT_local[(8)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(9)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(10)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(11)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(12)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(13)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(14)] = 0.000000e+00f;
  T_batch_matmul_NT_local[(15)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    ((float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(placeholder + (((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 64))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 512))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 128))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 1024))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 192))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 1536))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 256))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 2048))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 320))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 2560))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 384))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 3072))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 448))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 3584))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 512))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 4096))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 576))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 4608))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 640))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 5120))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 704))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 5632))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 768))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 6144))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 832))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 6656))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 896))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 7168))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 960))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 7680))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 1024))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 16384) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 8192))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1088) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 64))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1088) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 512))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1152) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 128))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1152) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 1024))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1216) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 192))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1216) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 1536))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1280) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 256))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1280) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 2048))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1344) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 320))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1344) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 2560))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1408) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 384))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1408) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 3072))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1472) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 448))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1472) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 3584))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1536) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 512))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1536) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 4096))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1600) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 576))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1600) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 4608))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1664) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 640))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1664) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 5120))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1728) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 704))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1728) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 5632))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1792) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 768))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1792) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 6144))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1856) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 832))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1856) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 6656))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1920) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 896))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1920) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 7168))))[0];
    ((float2*)(placeholder_d_shared + (((((((((int)threadIdx.x) * 2) + 1984) >> 10) * 1024) + (((int)threadIdx.x) * 2)) + 960))))[0] = ((float2*)(placeholder + (((((((((int)blockIdx.x) >> 6) * 16384) + ((((((int)threadIdx.x) * 2) + 1984) >> 10) * 8192)) + (k_outer_outer * 64)) + (((int)threadIdx.x) * 2)) + 7680))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(placeholder1 + (((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1024))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3072))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 5120))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 7168))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((int)blockIdx.x) & 63) * 8192)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 524288))))[0];
    ((float4*)(placeholder_shared + (((((((((int)threadIdx.x) * 4) + 1152) >> 10) * 1024) + (((((int)threadIdx.x) >> 4) + 2) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((((int)threadIdx.x) * 4) + 1152) >> 10) * 524288)) + ((((int)blockIdx.x) & 63) * 8192)) + (((((int)threadIdx.x) >> 4) + 2) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((((((int)threadIdx.x) * 4) + 1280) >> 10) * 1024) + (((((int)threadIdx.x) >> 4) + 4) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((((int)threadIdx.x) * 4) + 1280) >> 10) * 524288)) + ((((int)blockIdx.x) & 63) * 8192)) + (((((int)threadIdx.x) >> 4) + 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((((((int)threadIdx.x) * 4) + 1408) >> 10) * 1024) + (((((int)threadIdx.x) >> 4) + 6) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((((int)threadIdx.x) * 4) + 1408) >> 10) * 524288)) + ((((int)blockIdx.x) & 63) * 8192)) + (((((int)threadIdx.x) >> 4) + 6) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((((((int)threadIdx.x) * 4) + 1536) >> 10) * 1024) + (((((int)threadIdx.x) >> 4) + 8) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((((int)threadIdx.x) * 4) + 1536) >> 10) * 524288)) + ((((int)blockIdx.x) & 63) * 8192)) + (((((int)threadIdx.x) >> 4) + 8) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((((((int)threadIdx.x) * 4) + 1664) >> 10) * 1024) + (((((int)threadIdx.x) >> 4) + 10) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((((int)threadIdx.x) * 4) + 1664) >> 10) * 524288)) + ((((int)blockIdx.x) & 63) * 8192)) + (((((int)threadIdx.x) >> 4) + 10) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((((((int)threadIdx.x) * 4) + 1792) >> 10) * 1024) + (((((int)threadIdx.x) >> 4) + 12) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((((int)threadIdx.x) * 4) + 1792) >> 10) * 524288)) + ((((int)blockIdx.x) & 63) * 8192)) + (((((int)threadIdx.x) >> 4) + 12) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((((((int)threadIdx.x) * 4) + 1920) >> 10) * 1024) + (((((int)threadIdx.x) >> 4) + 14) * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((float4*)(placeholder1 + ((((((((((int)blockIdx.x) >> 6) * 1048576) + ((((((int)threadIdx.x) * 4) + 1920) >> 10) * 524288)) + ((((int)blockIdx.x) & 63) * 8192)) + (((((int)threadIdx.x) >> 4) + 14) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 8; ++k_inner) {
        T_batch_matmul_NT_local[(0)] = (T_batch_matmul_NT_local[(0)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner))] * placeholder_shared[((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner))]));
        T_batch_matmul_NT_local[(1)] = (T_batch_matmul_NT_local[(1)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 64))]));
        T_batch_matmul_NT_local[(2)] = (T_batch_matmul_NT_local[(2)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 128))]));
        T_batch_matmul_NT_local[(3)] = (T_batch_matmul_NT_local[(3)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 192))]));
        T_batch_matmul_NT_local[(4)] = (T_batch_matmul_NT_local[(4)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 256))]));
        T_batch_matmul_NT_local[(5)] = (T_batch_matmul_NT_local[(5)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 320))]));
        T_batch_matmul_NT_local[(6)] = (T_batch_matmul_NT_local[(6)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 384))]));
        T_batch_matmul_NT_local[(7)] = (T_batch_matmul_NT_local[(7)] + (placeholder_d_shared[(((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 448))]));
        T_batch_matmul_NT_local[(8)] = (T_batch_matmul_NT_local[(8)] + (placeholder_d_shared[((((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner) + 64))] * placeholder_shared[((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner))]));
        T_batch_matmul_NT_local[(9)] = (T_batch_matmul_NT_local[(9)] + (placeholder_d_shared[((((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner) + 64))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 64))]));
        T_batch_matmul_NT_local[(10)] = (T_batch_matmul_NT_local[(10)] + (placeholder_d_shared[((((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner) + 64))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 128))]));
        T_batch_matmul_NT_local[(11)] = (T_batch_matmul_NT_local[(11)] + (placeholder_d_shared[((((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner) + 64))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 192))]));
        T_batch_matmul_NT_local[(12)] = (T_batch_matmul_NT_local[(12)] + (placeholder_d_shared[((((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner) + 64))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 256))]));
        T_batch_matmul_NT_local[(13)] = (T_batch_matmul_NT_local[(13)] + (placeholder_d_shared[((((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner) + 64))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 320))]));
        T_batch_matmul_NT_local[(14)] = (T_batch_matmul_NT_local[(14)] + (placeholder_d_shared[((((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner) + 64))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 384))]));
        T_batch_matmul_NT_local[(15)] = (T_batch_matmul_NT_local[(15)] + (placeholder_d_shared[((((((((int)threadIdx.x) >> 1) * 128) + (k_outer_inner * 8)) + k_inner) + 64))] * placeholder_shared[(((((((((int)threadIdx.x) >> 4) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + (k_outer_inner * 8)) + k_inner) + 448))]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int j_inner = 0; j_inner < 8; ++j_inner) {
      T_batch_matmul_NT[((((((((((int)blockIdx.x) >> 6) * 32768) + ((((int)threadIdx.x) >> 1) * 2048)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 63) * 16)) + ((((int)threadIdx.x) & 1) * 8)) + j_inner))] = T_batch_matmul_NT_local[(((i_inner * 8) + j_inner))];
    }
  }
}

int main(int argc, char const* argv[]) {
  return 0;
}
