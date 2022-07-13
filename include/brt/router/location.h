/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once

#include <brt/runtime/cuda_utils.h>

namespace brt {
namespace router {

void GenerateGlobalDSTIndices(int* hot_mask /*[sample_num x path_num]*/,
                              int* global_dst_indices /*[sample_num x path_num]*/,
                              int* loads /*[path_num]*/, int* dst_indices_base /*[path_num]*/,
                              int* supported_capacities /*[supported_capacity_num]*/,
                              int sample_num, int path_num, int supported_capacity_num,
                              cudaStream_t stream);

void GenerateDSTIndices(int* hot_mask /*[sample_num x path_num]*/,
                        int* dst_indices /*[sample_num x path_num]*/, int* loads /*[path_num]*/,
                        int* supported_capacities /*[supported_capacity_num]*/, int sample_num,
                        int path_num, int supported_capacity_num, cudaStream_t stream);

void GenerateSRCIndices(int* hot_mask /*[sample_num x path_num]*/,
                        int* src_indices /*[sample_num x path_num]*/, int* loads /*[path_num]*/,
                        int* supported_capacities /*[supported_capacity_num]*/, int sample_num,
                        int path_num, int supported_capacity_num, cudaStream_t stream);

void CoordinateIndexFormat(int* origin_indices, int* new_indices, int* loads, int sample_num,
                          int path_num, int target_index_fmt_id, cudaStream_t stream);

}  // namespace router
}  // namespace brt
