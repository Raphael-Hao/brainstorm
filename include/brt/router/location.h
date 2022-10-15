/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef INCLUDE_BRT_ROUTER_LOCATION_H_
#define INCLUDE_BRT_ROUTER_LOCATION_H_

#include <brt/runtime/cuda_utils.h>

namespace brt {
namespace router {

void GenerateGlobalDstIndices(int* hot_mask /*[sample_num x path_num]*/,
                              int* global_dst_indices /*[sample_num x path_num]*/,
                              int* loads /*[path_num]*/, int* dst_indices_base /*[path_num]*/,
                              int* supported_capacities /*[supported_capacity_num]*/,
                              const int& sample_num, const int& path_num,
                              const int& supported_capacity_num, cudaStream_t stream);

void GenerateDstIndices(int* hot_mask /*[sample_num x path_num]*/,
                        int* dst_indices /*[sample_num x path_num]*/, int* loads /*[path_num]*/,
                        int* supported_capacities /*[supported_capacity_num]*/,
                        const int& sample_num, const int& path_num,
                        const int& supported_capacity_num, cudaStream_t stream);

void GenerateSrcIndices(int* hot_mask /*[sample_num x path_num]*/,
                        int* src_indices /*[sample_num x path_num]*/, int* loads /*[path_num]*/,
                        int* supported_capacities /*[supported_capacity_num]*/,
                        const int& sample_num, const int& path_num,
                        const int& supported_capacity_num, cudaStream_t stream);

void ConvertIndexFormat(int* origin_indices, int* new_indices, int* loads, const int& sample_num,
                        const int& path_num, const int& target_index_fmt_id, cudaStream_t stream);

}  // namespace router
}  // namespace brt
#endif  // INCLUDE_BRT_ROUTER_LOCATION_H_
