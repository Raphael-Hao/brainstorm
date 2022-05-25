/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once

#include <brt/runtime/cuda_utils.h>

namespace brt {
namespace router {
void MakeLocationAndLoad(int* one_hot_mask, int* locations, int* branch_loads,
                         int* supported_capacities, int num_samples, int branch_num,
                         int supported_capacity_num, cudaStream_t stream);

}  // namespace router
}  // namespace brt
