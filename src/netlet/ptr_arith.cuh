/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

template <typename T>
__global__ void __launch_bounds__(32) ptr_to_ptr_array(T** dst, T* src, int index[]) {
  // [thread_extent] blockIdx.xdim = length of dst and index
  // [thread_extent] blockIdx.ydim = 1
  // [thread_extent] blockIdx.zdim = 1
  // [thread_extent] threadIdx.xdim = 32
  // [thread_extent] threadIdx.ydim = 1
  // [thread_extent] threadIdx.zdim = 1
  dst[((blockIdx.x * 32) + threadIdx.x)] =
      src + index[((blockIdx.x * 32) + threadIdx.x)] * sizeof(T);
}

template <typename T>
__global__ void __launch_bounds__(32) ptrs_to_ptr_array(T** dst, T* src, int index[]) {
  // [thread_extent] blockIdx.xdim = length of dst and index
  // [thread_extent] blockIdx.ydim = 1
  // [thread_extent] blockIdx.zdim = 1
  // [thread_extent] threadIdx.xdim = 32
  // [thread_extent] threadIdx.ydim = 1
  // [thread_extent] threadIdx.zdim = 1
  dst[((blockIdx.x * 32) + threadIdx.x)] =
      src + index[((blockIdx.x * 32) + threadIdx.x)] * sizeof(T);
}