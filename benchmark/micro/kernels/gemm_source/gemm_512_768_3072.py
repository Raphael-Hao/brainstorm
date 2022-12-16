# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

dependency = r"""
extern "C" __device__ __forceinline__
uint smem_u32addr(const void *smem_ptr) {
    uint addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

extern "C" __device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

extern "C" __device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

extern "C" __device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

extern "C" __device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

extern "C" __device__ __forceinline__
void sts32(const float &reg, const uint &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

extern "C" __device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

extern "C" struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

extern "C" __device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint C_sts_addr,
               uint m,
               uint n,
               uint m_idx,
               uint n_idx) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}
"""

# placeholder 512 768 768 3072 512 768

source_code = r"""
extern "C" __global__ void __launch_bounds__(256) sgemm_512x768x3072(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_matmul_NT) {
  // [kernel_type] global
  // [thread_extent] blockIdx.x = 24
  // [thread_extent] blockIdx.y = 4
  // [thread_extent] blockIdx.z = 1
  // [thread_extent] threadIdx.x = 256
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] threadIdx.z = 1

   __shared__ char smem[24576];

   float *A_smem = reinterpret_cast<float *>(smem);
   float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);

   // placeholder, placeholder1 and T_matmul_NT register fragment
   float A_frag[2][8];
   float B_frag[2][8];
   float C_frag[8][8];
#pragma unroll
   for (int i = 0; i < 8; ++i) {
#pragma unroll
     for (int j = 0; j < 8; ++j) {
       C_frag[i][j] = 0;
     }
   }

   const uint lane_id = threadIdx.x % 32;
   const uint warp_id = threadIdx.x / 32;

   // 4x8 threads each warp for FFMA
   const uint mma_tid_x = (lane_id / 2) % 8;
   const uint mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

   // A_tile & B_tile ldg pointer
   const char *A_ldg_ptr =
       (const char *)(placeholder +
                      (blockIdx.y * 128 + threadIdx.x / 8 * 4) * 768 +
                      threadIdx.x % 8);

   // const char *B_ldg_ptr = (const char *)(
   //     placeholder1 + (threadIdx.x / 32) * 3072 + blockIdx.x * 128 +
   //     threadIdx.x % 32);
   const char *B_ldg_ptr =
       (const char *)(placeholder1 +
                      (blockIdx.x * 128 + threadIdx.x / 8 * 4) * 768 +
                      threadIdx.x % 8);

   // A_tile & B_tile sts/lds pointer
   // using uint pointer for faster double buffer switch
   uint A_sts_addr =
       smem_u32addr(A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
   // uint B_sts_addr = smem_u32addr(
   //     B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));
   uint B_sts_addr =
       smem_u32addr(B_smem + (threadIdx.x % 8) * 128 + (threadIdx.x / 8) * 4);
   ;

   uint A_lds_addr = smem_u32addr(A_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
   uint B_lds_addr = smem_u32addr(B_smem + (warp_id % 2) * 64 + mma_tid_x * 4);

   // ldg_guard to avoid LDG out of bound
   uint A_ldg_guard = 0;
#pragma unroll
   for (int i = 0; i < 4; ++i) {
     int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
     if (m_idx < 512) {
       A_ldg_guard |= (1u << i);
     }
   }

   uint B_ldg_guard = 0;
#pragma unroll
   for (int i = 0; i < 4; ++i) {
     int n_idx = blockIdx.x * 128 + threadIdx.x / 8 * 4 + i;
     if (n_idx < 3072) {
       B_ldg_guard |= (1u << i);
     }
   }

   float A_ldg_reg[4];
   float B_ldg_reg[4];

   // 1'st placeholder&placeholder1 tile loaded before the k_tile loop
   uint k_tiles = (768 + 7) / 8 - 1;

   // load 1'st tile to shared memory
   {
     uint first_k_tile = 768 - k_tiles * 8;

#pragma unroll
     for (int i = 0; i < 4; ++i) {
       bool guard =
           (A_ldg_guard & (1u << i)) != 0 && threadIdx.x % 8 < first_k_tile;
       ldg32_nc_0(A_ldg_reg[i], A_ldg_ptr + i * 3072, guard);
     }

     sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3], A_sts_addr);

#pragma unroll
     for (int i = 0; i < 4; ++i) {
       bool guard =
           (B_ldg_guard & (1u << i)) != 0 && threadIdx.x % 8 < first_k_tile;
       ldg32_nc_0(B_ldg_reg[i], B_ldg_ptr + i * 3072, guard);
     }
     sts128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3], B_sts_addr);

     __syncthreads();

     // switch double buffer
     A_sts_addr ^= 0x2000;
     B_sts_addr ^= 0x1000;

     // ldg pointer for next tile
     A_ldg_ptr += first_k_tile * sizeof(float);
     B_ldg_ptr += first_k_tile * sizeof(float);
   }

   // load 1'st fragment
   lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
   lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
          A_lds_addr + 16 * sizeof(float));
   lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
   lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
          B_lds_addr + 32 * sizeof(float));

   // k_tiles loop
   for (; k_tiles > 0; --k_tiles) {
#pragma unroll
     for (int k_frag = 0; k_frag < 8; ++k_frag) {
       // store next placeholder&placeholder1 tile to shared memory
       if (k_frag == 7) {
         sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                A_sts_addr);

         sts128(B_ldg_reg[0], B_ldg_reg[1], B_ldg_reg[2], B_ldg_reg[3],
                B_sts_addr);
         // #pragma unroll
         // for (int i = 0; i < 4; ++i) {
         //     sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
         // }

         __syncthreads();

         // switch double buffer
         A_lds_addr ^= 0x2000;
         B_lds_addr ^= 0x1000;
         A_sts_addr ^= 0x2000;
         B_sts_addr ^= 0x1000;

         // ldg pointer for next tile
         A_ldg_ptr += 8 * sizeof(float);
         // B_ldg_ptr += 98304;
         B_ldg_ptr += 8 * sizeof(float);
       }

       // load next placeholder&placeholder1 fragment from shared memory to
       // register
       lds128(A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
              A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
              A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
       lds128(A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
              A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
              A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
       lds128(B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
              B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
              B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
       lds128(B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
              B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
              B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));

       // load next placeholder&placeholder1 tile
       if (k_frag == 0) {
#pragma unroll
         for (int i = 0; i < 4; ++i) {
           ldg32_nc(A_ldg_reg[i], A_ldg_ptr + i * 3072,
                    (A_ldg_guard & (1u << i)) != 0);
         }

#pragma unroll
         for (int i = 0; i < 4; ++i) {
           ldg32_nc(B_ldg_reg[i], B_ldg_ptr + i * 3072,
                    (B_ldg_guard & (1u << i)) != 0);
         }
       }

// FFMA loop
#pragma unroll
       for (int i = 0; i < 8; ++i) {
#pragma unroll
         for (int j = 0; j < 8; ++j) {
           C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
         }
       }
     }
   }

// FFMA for the last tile
#pragma unroll
   for (int k_frag = 0; k_frag < 8; ++k_frag) {
     if (k_frag < 7) {
       // load next placeholder&placeholder1 fragment from shared memory to
       // register
       lds128(A_frag[(k_frag + 1) % 2][0], A_frag[(k_frag + 1) % 2][1],
              A_frag[(k_frag + 1) % 2][2], A_frag[(k_frag + 1) % 2][3],
              A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
       lds128(A_frag[(k_frag + 1) % 2][4], A_frag[(k_frag + 1) % 2][5],
              A_frag[(k_frag + 1) % 2][6], A_frag[(k_frag + 1) % 2][7],
              A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
       lds128(B_frag[(k_frag + 1) % 2][0], B_frag[(k_frag + 1) % 2][1],
              B_frag[(k_frag + 1) % 2][2], B_frag[(k_frag + 1) % 2][3],
              B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
       lds128(B_frag[(k_frag + 1) % 2][4], B_frag[(k_frag + 1) % 2][5],
              B_frag[(k_frag + 1) % 2][6], B_frag[(k_frag + 1) % 2][7],
              B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
     }

// FFMA loop
#pragma unroll
     for (int i = 0; i < 8; ++i) {
#pragma unroll
       for (int j = 0; j < 8; ++j) {
         C_frag[i][j] += A_frag[k_frag % 2][i] * B_frag[k_frag % 2][j];
       }
     }
   }

   // C_tile write back, reuse placeholder&placeholder1 tile shared memory
   // buffer
   uint C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                  mma_tid_y * 4 * 8 + mma_tid_x);
   const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;

   uint m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
   uint n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

   float *C_stg_ptr = T_matmul_NT + m_idx * 3072 + n_idx;

   if (m_idx >= 512) {
     return;
   } else if (m_idx + 32 <= 512) {
     uint n_guard = 3072 < n_idx ? 0 : 3072 - n_idx;

#pragma unroll
     for (int i = 0; i < 2; ++i) {
#pragma unroll
       for (int j = 0; j < 2; ++j) {
         __syncthreads();

#pragma unroll
         for (int p = 0; p < 4; ++p) {
           sts128(C_frag[i * 4 + p][j * 4], C_frag[i * 4 + p][j * 4 + 1],
                  C_frag[i * 4 + p][j * 4 + 2], C_frag[i * 4 + p][j * 4 + 3],
                  C_sts_addr + p * 8 * sizeof(float4));
         }

         __syncthreads();

#pragma unroll
         for (int p = 0; p < 16; ++p) {
           stg32(C_lds_ptr[p * 32], C_stg_ptr + (i * 16 + p) * 3072 + j * 32,
                 j * 32 < n_guard);
         }
       }
     }
   } else {
#pragma unroll
     for (int i = 0; i < 2; ++i) {
#pragma unroll
       for (int j = 0; j < 2; ++j) {
         StgFrag stg_frag(C_frag, j, i);

         C_tile_wb(stg_frag, C_stg_ptr + i * 16 * 3072 + j * 32, C_lds_ptr,
                   C_sts_addr, 512, 3072, m_idx + i * 16, n_idx + j * 32);
       }
     }
   }
}
"""
