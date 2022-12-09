/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_ROUTER_ROUTE_H_
#define BRT_ROUTER_ROUTE_H_

namespace brt {
namespace router {

void DispatchWithDstIndices1D(float* src_data /*[sample_num x sample_size]*/,
                              float* dst_data /*[?load*dst_num x sample_size]*/,
                              float* gates /*[sample_num x dst_num]*/,
                              int* route_indices /*[sample_num x dst_num]*/,
                              int* loads /*[dst_num]*/, const int& capacity, const int& sample_num,
                              const int& sample_size, const int& path_num, cudaStream_t stream);

void DispatchWithDstIndices2D(float* src_data /*[sample_num x sample_size]*/,
                              float* dst_data /*[?load*dst_num x sample_size]*/,
                              int* route_indices /*[sample_num x dst_num]*/,
                              int* loads /*[dst_num]*/, const int& capacity, const int& sample_num,
                              const int& sample_size, const int& path_num, cudaStream_t stream);

void CombineWithSrcIndices(float* src_data /*[?load*dst_num x sample_size]*/,
                           float* dst_data /*[sample_num x sample_size]*/,
                           float* gates /*[sample_num x dst_num]*/,
                           int* route_indices /*[sample_num x dst_num]*/, int* loads /*[dst_num]*/,
                           const int& capacity, const int& sample_num, const int& sample_size,
                           const int& path_num, cudaStream_t stream);
void ResidualCombineWithSrcIndices(float* src_data /*[?load*path_num x sample_size]*/,
                                   float* dst_data /*[sample_num x sample_size]*/,
                                   float* gates /*[sample_num x path_num]*/,
                                   int* route_indices /*[sample_num x path_num]*/,
                                   int* loads /*[path_num]*/, const int& capacity,
                                   const int& sample_num, const int& sample_size,
                                   const int& path_num, cudaStream_t stream);
}  // namespace router
}  // namespace brt

#endif  // BRT_ROUTER_ROUTE_H_
