# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
#%%
import re

re_func = re.compile(r'(.*__global__\s+void\s+[A-Za-z_]\w*\s*\([^{]*\))\s*({.+\Z)', re.DOTALL)

function = r"""
extern "C" __global__ void __launch_bounds__(64) default_function_kernel0(float* __restrict__ placeholder, float* __restrict__ placeholder1, float* __restrict__ T_add, float* __restrict__ placeholder2) {
  float T_matmul_NT[1];
  __shared__ float placeholder_d_shared[128];
  __shared__ float placeholder_shared[8192];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    *(float2*)(placeholder_d_shared + (((int)threadIdx.x) * 2)) = *(float2*)(placeholder + ((k_outer_outer * 128) + (((int)threadIdx.x) * 2)));
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 32; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(placeholder_shared + ((ax0_ax1_fused_outer_outer * 256) + (((int)threadIdx.x) * 4))) = *(float4*)(placeholder1 + (((((((int)blockIdx.x) * 131072) + (ax0_ax1_fused_outer_outer * 4096)) + ((((int)threadIdx.x) >> 5) * 2048)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 31) * 4)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 16; ++k_outer_inner) {
      for (int k_inner = 0; k_inner < 8; ++k_inner) {
        T_matmul_NT[0] = (T_matmul_NT[0] + (placeholder_d_shared[((k_outer_inner * 8) + k_inner)] * placeholder_shared[(((((int)threadIdx.x) * 128) + (k_outer_inner * 8)) + k_inner)]));
      }
    }
  }
  T_add[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (T_matmul_NT[0] + placeholder2[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))]);
}
"""

sig, body = re_func.match(function).groups()
print(sig)
print(body)
# %%
