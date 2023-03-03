/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_ROUTER_LOCATION_H_
#define BRT_ROUTER_LOCATION_H_

#include <cuda_runtime.h>
namespace brt {
namespace router {

void GenerateDstIndices(int* hot_mask /*[sample_num x path_num]*/,
                        int* dst_indices /*[sample_num x path_num]*/,
                        int* loads /*[path_num]*/,
                        int* supported_capacities /*[supported_capacity_num]*/,
                        const int& sample_num,
                        const int& path_num,
                        const int& supported_capacity_num,
                        cudaStream_t stream);

void GenerateSrcIndices(int* hot_mask /*[sample_num x path_num]*/,
                        int* src_indices /*[sample_num x path_num]*/,
                        int* loads /*[path_num]*/,
                        int* supported_capacities /*[supported_capacity_num]*/,
                        const int& sample_num,
                        const int& path_num,
                        const int& supported_capacity_num,
                        cudaStream_t stream);

void ConvertIndexFormat(int* origin_indices,
                        int* new_indices,
                        int* loads,
                        const int& sample_num,
                        const int& path_num,
                        const int& target_index_fmt_id,
                        cudaStream_t stream);

template <bool is_dst_index>
void GenerateIndicesAndLoads(int* hot_mask /*[sample_num x path_num]*/,
                             int* indices /*[sample_num x path_num]*/,
                             int* loads /*[path_num]*/,
                             const int& sample_num,
                             const int& path_num,
                             int* supported_capacities /*[supported_capacity_num]*/,
                             const int& supported_capacity_num,
                             const bool& capacity_padding,
                             cudaStream_t stream);

}  // namespace router
}  // namespace brt
#endif  // BRT_ROUTER_LOCATION_H_
