/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_ROUTER_ROUTE_H_
#define BRT_ROUTER_ROUTE_H_

#include <cuda_fp16.h>

namespace brt {
namespace router {

template <typename dtype>
void DispatchWithDstIndices1D(void* src_data /*[sample_num, sample_size]*/,
                              void* dst_data /*[total_loads, sample_size]*/,
                              void* gates /*[sample_num, dst_num]*/,
                              int* route_indices /*[sample_num, dst_num]*/,
                              int* loads /*[dst_num]*/, const int& capacity, const int& sample_num,
                              const int& sample_size, const int& path_num, cudaStream_t stream);

template <typename dtype>
void DispatchWithDstIndices2D(void* src_data /*[sample_num, sample_size]*/,
                              void* dst_data /*[total_loads, sample_size]*/,
                              int* route_indices /*[sample_num, dst_num]*/,
                              int* loads /*[dst_num]*/, const int& capacity, const int& sample_num,
                              const int& sample_size, const int& path_num, cudaStream_t stream);

template <typename dtype>
void CombineWithSrcIndices(void* src_data /*[total_loads, sample_size]*/,
                           void* dst_data /*[sample_num, sample_size]*/,
                           void* gates /*[sample_num, dst_num]*/,
                           int* route_indices /*[sample_num, dst_num]*/, int* loads /*[dst_num]*/,
                           const int& capacity, const int& sample_num, const int& sample_size,
                           const int& path_num, cudaStream_t stream);

template <typename dtype>
void ResidualCombineWithSrcIndices(void* src_data /*[total_loads, sample_size]*/,
                                   void* dst_data /*[sample_num, sample_size]*/,
                                   void* gates /*[sample_num x path_num]*/,
                                   int* route_indices /*[sample_num x path_num]*/,
                                   int* loads /*[path_num]*/, const int& capacity,
                                   const int& sample_num, const int& sample_size,
                                   const int& path_num, cudaStream_t stream);
}  // namespace router
}  // namespace brt

#endif  // BRT_ROUTER_ROUTE_H_
