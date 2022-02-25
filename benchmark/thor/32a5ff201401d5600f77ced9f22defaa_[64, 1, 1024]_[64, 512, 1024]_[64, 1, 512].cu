
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
extern "C" __global__ void __launch_bounds__(16)
    default_function_kernel0(float* __restrict__ placeholder,
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
