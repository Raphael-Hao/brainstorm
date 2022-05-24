/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once

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
    dst[global_tid] = src + index[global_tid] * granularity;
  }
}

void DevicePtr2PtrArray() {
  
}