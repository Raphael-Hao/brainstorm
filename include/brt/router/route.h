/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_ROUTER_ROUTE_H_
#define BRT_ROUTER_ROUTE_H_

#include <cuda_runtime.h>

namespace brt {
namespace router {

template <typename dtype>
void DispatchWithDstIndices1D(void* src_data /*[cell_num, cell_size]*/,
                              void* dst_data /*[total_loads, cell_size]*/,
                              void* gates /*[cell_num, dst_num]*/,
                              int* route_indices /*[cell_num, dst_num]*/,
                              int* loads /*[dst_num]*/,
                              const int& cell_num_per_path,
                              const int& cell_num,
                              const int& cell_size,
                              const int& path_num,
                              cudaStream_t stream);

template <typename dtype>
void DispatchWithDstIndices2D(void* src_data /*[cell_num, cell_size]*/,
                              void* dst_data /*[total_loads, cell_size]*/,
                              int* route_indices /*[cell_num, dst_num]*/,
                              int* loads /*[dst_num]*/,
                              const int& cell_num_per_path,
                              const int& cell_num,
                              const int& cell_size,
                              const int& path_num,
                              cudaStream_t stream);

template <typename dtype>
void CombineWithSrcIndices(void* src_data /*[total_loads, cell_size]*/,
                           void* dst_data /*[cell_num, cell_size]*/,
                           void* gates /*[cell_num, dst_num]*/,
                           int* route_indices /*[cell_num, dst_num]*/,
                           int* loads /*[dst_num]*/,
                           const int& cell_num_per_path,
                           const int& cell_num,
                           const int& cell_size,
                           const int& path_num,
                           cudaStream_t stream);

template <typename dtype>
void ResidualCombineWithSrcIndices(void* src_data /*[total_loads, cell_size]*/,
                                   void* dst_data /*[cell_num, cell_size]*/,
                                   void* gates /*[cell_num x path_num]*/,
                                   int* route_indices /*[cell_num x path_num]*/,
                                   int* loads /*[path_num]*/,
                                   const int& cell_num_per_path,
                                   const int& cell_num,
                                   const int& cell_size,
                                   const int& path_num,
                                   cudaStream_t stream);

template <typename dtype>
void DispatchWithIndicesAndLoads(void* src_data /*[cell_num, cell_size]*/,
                                 void* dst_data /*[total_loads, cell_size]*/,
                                 void* gates /*[cell_num, dst_num]*/,
                                 int* route_indices /*[cell_num, dst_num]*/,
                                 int* loads /*[dst_num]*/,
                                 const int& cell_num,
                                 const int& cell_size,
                                 const int& path_num,
                                 const int& cell_num_per_path,
                                 bool is_1d_routing,
                                 bool is_dst_index,
                                 cudaStream_t stream);

template <typename dtype>
void CombineWithIndicesAndLoads(void* src_data /*[total_loads, cell_size]*/,
                                void* dst_data /*[cell_num, cell_size]*/,
                                void* gates /*[cell_num, dst_num]*/,
                                int* route_indices /*[cell_num, dst_num]*/,
                                int* loads /*[dst_num]*/,
                                const int& cell_num,
                                const int& cell_size,
                                const int& path_num,
                                const int& cell_num_per_path,
                                bool is_residual,
                                bool is_dst_index,
                                cudaStream_t stream);
}  // namespace router
}  // namespace brt

#endif  // BRT_ROUTER_ROUTE_H_
