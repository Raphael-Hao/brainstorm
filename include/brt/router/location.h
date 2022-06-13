/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once

#include <brt/runtime/cuda_utils.h>

namespace brt {
namespace router {
void GenerateGlobalRouteIndices(int* hot_mask /*[sample_num x dst_num]*/,
                                int* route_indices /*[sample_num x dst_num]*/,
                                int* dst_loads /*[dst_num]*/, int* dst_start_indices /*[dst_num]*/,
                                int* supported_capacities /*[supported_capacity_num]*/,
                                int sample_num, int dst_num, int supported_capacity_num,
                                cudaStream_t stream);

void GenerateLocalRouteIndices(int* hot_mask /*[sample_num x dst_num]*/,
                               int* route_indices /*[sample_num x dst_num]*/,
                               int* dst_loads /*[dst_num]*/,
                               int* supported_capacities /*[supported_capacity_num]*/,
                               int sample_num, int dst_num, int supported_capacity_num,
                               cudaStream_t stream);

}  // namespace router
}  // namespace brt
