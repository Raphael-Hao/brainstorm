
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
  __shared__ float placeholder_d_shared[256];
  __shared__ float placeholder_shared[4096];
  T_batch_matmul_NT_local[(0)] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    ((float2*)(placeholder_d_shared + ((((int)threadIdx.x) * 2))))[0] = ((
        float2*)(placeholder +
                 (((((((int)blockIdx.x) >> 6) * 512) + (k_outer_outer * 256)) +
                   (((int)threadIdx.x) * 2)))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 32))))[0] =
        ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 512) +
                                     (k_outer_outer * 256)) +
                                    (((int)threadIdx.x) * 2)) +
                                   32))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 64))))[0] =
        ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 512) +
                                     (k_outer_outer * 256)) +
                                    (((int)threadIdx.x) * 2)) +
                                   64))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 96))))[0] =
        ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 512) +
                                     (k_outer_outer * 256)) +
                                    (((int)threadIdx.x) * 2)) +
                                   96))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 128))))[0] =
        ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 512) +
                                     (k_outer_outer * 256)) +
                                    (((int)threadIdx.x) * 2)) +
                                   128))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 160))))[0] =
        ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 512) +
                                     (k_outer_outer * 256)) +
                                    (((int)threadIdx.x) * 2)) +
                                   160))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 192))))[0] =
        ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 512) +
                                     (k_outer_outer * 256)) +
                                    (((int)threadIdx.x) * 2)) +
                                   192))))[0];
    ((float2*)(placeholder_d_shared + (((((int)threadIdx.x) * 2) + 224))))[0] =
        ((float2*)(placeholder + ((((((((int)blockIdx.x) >> 6) * 512) +
                                     (k_outer_outer * 256)) +
                                    (((int)threadIdx.x) * 2)) +
                                   224))))[0];
    ((float4*)(placeholder_shared + ((((int)threadIdx.x) * 4))))[0] =
        ((float4*)(placeholder1 +
                   ((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                     (((int)threadIdx.x) * 4)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 64))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     64))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 128))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     128))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 192))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     192))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 256))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     512))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 320))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 320) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 384))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 384) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 448))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 448) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 512))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     1024))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 576))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 576) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 640))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 640) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 704))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 704) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 768))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     1536))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 832))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 832) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 896))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 896) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 960))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 960) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     2048))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1088))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1088) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1152))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1152) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1216))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1216) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1280))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     2560))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1344))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1344) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1408))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1408) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1472))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1472) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     3072))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1600))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1600) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1664))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1664) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1728))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1728) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1792))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     3584))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1856))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1856) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1920))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1920) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 1984))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 1984) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2048))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     4096))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2112))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2112) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2176))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2176) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2240))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2240) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2304))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     4608))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2368))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2368) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2432))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2432) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2496))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2496) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2560))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     5120))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2624))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2624) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2688))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2688) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2752))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2752) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2816))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     5632))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2880))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2880) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 2944))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 2944) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3008))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3008) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3072))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     6144))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3136))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3136) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3200))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3200) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3264))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3264) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3328))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     6656))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3392))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3392) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3456))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3456) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3520))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3520) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3584))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     7168))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3648))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3648) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3712))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3712) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3776))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3776) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3840))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) + (k_outer_outer * 256)) +
                      (((int)threadIdx.x) * 4)) +
                     7680))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3904))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3904) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 64)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 3968))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 3968) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 128)))))[0];
    ((float4*)(placeholder_shared + (((((int)threadIdx.x) * 4) + 4032))))[0] =
        ((float4*)(placeholder1 +
                   (((((((int)blockIdx.x) * 8192) +
                       ((((((int)threadIdx.x) * 4) + 4032) >> 8) * 512)) +
                      (k_outer_outer * 256)) +
                     ((((int)threadIdx.x) * 4) + 192)))))[0];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 64; ++k_outer_inner) {
      T_batch_matmul_NT_local[(0)] =
          (T_batch_matmul_NT_local[(0)] +
           (placeholder_d_shared[((k_outer_inner * 4))] *
            placeholder_shared[(
                ((((int)threadIdx.x) * 256) + (k_outer_inner * 4)))]));
      T_batch_matmul_NT_local[(0)] =
          (T_batch_matmul_NT_local[(0)] +
           (placeholder_d_shared[(((k_outer_inner * 4) + 1))] *
            placeholder_shared[(
                (((((int)threadIdx.x) * 256) + (k_outer_inner * 4)) + 1))]));
      T_batch_matmul_NT_local[(0)] =
          (T_batch_matmul_NT_local[(0)] +
           (placeholder_d_shared[(((k_outer_inner * 4) + 2))] *
            placeholder_shared[(
                (((((int)threadIdx.x) * 256) + (k_outer_inner * 4)) + 2))]));
      T_batch_matmul_NT_local[(0)] =
          (T_batch_matmul_NT_local[(0)] +
           (placeholder_d_shared[(((k_outer_inner * 4) + 3))] *
            placeholder_shared[(
                (((((int)threadIdx.x) * 256) + (k_outer_inner * 4)) + 3))]));
    }
  }
  T_batch_matmul_NT[(((((int)blockIdx.x) * 16) + ((int)threadIdx.x)))] =
      T_batch_matmul_NT_local[(0)];
}
