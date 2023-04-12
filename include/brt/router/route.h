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
void DispatchWithIndicesAndLoads(void* src_data /*[cell_num, cell_size]*/,
                                 void* dst_data /*[total_loads, cell_size]*/,
                                 void* gates /*[cell_num, dst_num]*/,
                                 int* route_indices /*[cell_num, dst_num]*/,
                                 int* loads /*[dst_num]*/,
                                 int* old_tags,
                                 int* new_tags,
                                 const int& cell_num,
                                 const int& cell_size,
                                 const int& path_num,
                                 const int& max_path_load,
                                 bool is_1d_routing,
                                 bool is_tag_index,
                                 cudaStream_t stream);

template <typename dtype>
void CombineWithIndicesAndLoads(void* src_data /*[total_loads, cell_size]*/,
                                void* dst_data /*[cell_num, cell_size]*/,
                                void* gates /*[cell_num, dst_num]*/,
                                int* route_indices /*[cell_num, dst_num]*/,
                                int* loads /*[dst_num]*/,
                                int* old_tags,
                                int* new_tags,
                                const int& cell_num,
                                const int& cell_size,
                                const int& path_num,
                                const int& max_path_load,
                                bool is_residual,
                                bool is_tag_index,
                                cudaStream_t stream);
}  // namespace router
}  // namespace brt

#endif  // BRT_ROUTER_ROUTE_H_
