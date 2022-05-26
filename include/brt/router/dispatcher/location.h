/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once

#include <brt/runtime/cuda_utils.h>

namespace brt {
namespace router {
void GenerateIndicesWithLoadMap(int* one_hot_mask, int* locations, int* branch_loads,
                                 int* branch_start_indices, int* supported_capacities,
                                 int sample_num, int branch_num, int supported_capacity_num,
                                 cudaStream_t stream);

}  // namespace router
}  // namespace brt
