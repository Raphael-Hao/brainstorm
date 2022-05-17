/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <thrust/host_vector.h>

template <typename T>
__global__ void __launch_bounds__(32)
    ptr_to_ptr_array(T** __restrict__ dst, T* __restrict__ src, int index[], int array_size,
                     int granularity) {
  // [thread_extent] blockIdx.xdim = length of dst and index
  // [thread_extent] blockIdx.ydim = 1
  // [thread_extent] blockIdx.zdim = 1
  // [thread_extent] threadIdx.xdim = 32
  // [thread_extent] threadIdx.ydim = 1
  // [thread_extent] threadIdx.zdim = 1
  int global_tid = blockIdx.x * 32 + threadIdx.x;
  if (global_tid < array_size) {
    dst[global_tid] = src + index[global_tid] * granularity;
  }
}

// template <typename T>
// void ptrs_to_ptr_array(thrust::host_vector<void*>& host_vector, T* src, int index[]) {
//   // [thread_extent] blockIdx.xdim = length of dst and index
//   // [thread_extent] blockIdx.ydim = 1
//   // [thread_extent] blockIdx.zdim = 1
//   // [thread_extent] threadIdx.xdim = 32
//   // [thread_extent] threadIdx.ydim = 1
//   // [thread_extent] threadIdx.zdim = 1
//   dst[((blockIdx.x * 32) + threadIdx.x)] =
//       src + index[((blockIdx.x * 32) + threadIdx.x)] * sizeof(T);
// }