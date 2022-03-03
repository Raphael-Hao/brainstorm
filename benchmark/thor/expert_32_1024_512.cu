#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long

extern "C" __global__ void __launch_bounds__(32)
    default_function_kernel0(float* __restrict__ placeholder,
                             float* __restrict__ placeholder1,
                             float* __restrict__ T_batch_matmul_NT) {
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
int main(int argc, char const* argv[]) {
  return 0;
}