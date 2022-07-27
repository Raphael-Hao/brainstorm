/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_ROUTER_ROUTE_H_
#define BRT_ROUTER_ROUTE_H_

#include <brt/runtime/cuda_utils.h>

namespace brt {
namespace router {

void DispatchWithDstIndices1D(float* src_data /*[sample_num x sample_dim]*/,
                              float* dst_data /*[?load*dst_num x sample_dim]*/,
                              float* gates /*[sample_num x dst_num]*/,
                              int* route_indices /*[sample_num x dst_num]*/,
                              int* loads /*[dst_num]*/, const int& sample_num,
                              const int& sample_dim, const int& path_num, cudaStream_t stream);

void DispatchWithDstIndices2D(float* src_data /*[sample_num x sample_dim]*/,
                              float* dst_data /*[?load*dst_num x sample_dim]*/,
                              int* route_indices /*[sample_num x dst_num]*/,
                              int* loads /*[dst_num]*/, const int& sample_num,
                              const int& sample_dim, const int& path_num, cudaStream_t stream);

void CombineWithSrcIndices(float* src_data /*[?load*dst_num x sample_dim]*/,
                           float* dst_data /*[sample_num x sample_dim]*/,
                           float* gates /*[sample_num x dst_num]*/,
                           int* route_indices /*[sample_num x dst_num]*/, int* loads /*[dst_num]*/,
                           const int& sample_num, const int& sample_dim, const int& path_num,
                           cudaStream_t stream);
}  // namespace router
}  // namespace brt

#endif  // BRT_ROUTER_ROUTE_H_
