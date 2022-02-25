
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
extern "C" __global__ void __launch_bounds__(32) default_function_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NT) {
  float T_matmul_NT_local[4];
  __shared__ float placeholder_d_shared[64];
  __shared__ float placeholder_shared[512];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int i_c_inner_init = 0; i_c_inner_init < 2; ++i_c_inner_init) {
      T_matmul_NT_local[(((i_c_outer_inner_init * 2) + i_c_inner_init))] = 0.000000e+00f;
    }
  }
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    ((float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 4) * 2048) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0];
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 8; ++ax0_ax1_fused_outer_outer) {
      ((float2*)(placeholder_shared + (((ax0_ax1_fused_outer_outer * 64) + (((int)threadIdx.x) * 2)))))[0] = ((float2*)(placeholder1 + (((((((((int)blockIdx.x) & 15) * 16384) + (ax0_ax1_fused_outer_outer * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
        for (int i_c_inner = 0; i_c_inner < 2; ++i_c_inner) {
          T_matmul_NT_local[(((i_c_outer_inner * 2) + i_c_inner))] = (T_matmul_NT_local[(((i_c_outer_inner * 2) + i_c_inner))] + (placeholder_d_shared[((((i_c_outer_inner * 32) + (i_c_inner * 16)) + k_outer_inner))] * placeholder_shared[(((((int)threadIdx.x) * 16) + k_outer_inner))]));
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 4; ++i_inner) {
    T_matmul_NT[((((((((int)blockIdx.x) >> 4) * 2048) + (i_inner * 512)) + ((((int)blockIdx.x) & 15) * 32)) + ((int)threadIdx.x)))] = T_matmul_NT_local[(i_inner)];
  }
}

