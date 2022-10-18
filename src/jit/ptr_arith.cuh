/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#ifndef BRT_JIT_PTR_ARITH_CUH_
#define BRT_JIT_PTR_ARITH_CUH_

namespace brt {
namespace jit {

template <typename T>
__global__ void __launch_bounds__(32)
    ptr_to_ptr_array(T** __restrict__ dst, T* __restrict__ src, int index[], int array_size,
                     int granularity) {
  // [thread_extent] blockIdx.x = length of dst and index
  // [thread_extent] blockIdx.y = 1
  // [thread_extent] blockIdx.z = 1
  // [thread_extent] threadIdx.x = 32
  // [thread_extent] threadIdx.y = 1
  // [thread_extent] threadIdx.z = 1
  int global_tid = blockIdx.x * 32 + threadIdx.x;
  if (global_tid < array_size) {
    dst[global_tid] = src + index[global_tid] * granularity * sizeof(T);
    // printf("pointer array id: %d, pointer: %p\n", global_tid, dst[global_tid]);
  }
}

template <typename T>
void DevicePtr2PtrArray(T** dst, T* src, int index[], int array_size, int granularity,
                        cudaStream_t stream) {
  const dim3 block_dim(32);
  const dim3 grid_dim((array_size + 31) / 32);
  ptr_to_ptr_array<T><<<grid_dim, block_dim, 0, stream>>>(dst, src, index, array_size, granularity);
}

}  // namespace jit
}  // namespace brt

#endif  // BRT_NETLET_PTR_ARITH_CUH_


#endif  // BRT_JIT_PTR_ARITH_CUH_
