/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_ROUTER_LOCATION_H_
#define BRT_ROUTER_LOCATION_H_

#include <cuda_runtime.h>
namespace brt {
namespace router {

void ConvertIndexFormat(int* origin_indices,
                        int* new_indices,
                        int* loads,
                        const int& cell_num,
                        const int& path_num,
                        const bool& tag_to_seat,
                        cudaStream_t stream);

void GenerateIndicesAndLoads(int* hot_mask /*[cell_num x path_num]*/,
                             int* indices /*[cell_num x path_num]*/,
                             int* loads /*[path_num]*/,
                             const int& cell_num,
                             const int& path_num,
                             int* supported_capacities /*[supported_capacity_num]*/,
                             const int& supported_capacity_num,
                             const bool& capacity_padding,
                             const bool& path_wise_padding,
                             const bool& is_tag_index,
                             cudaStream_t stream);

}  // namespace router
}  // namespace brt
#endif  // BRT_ROUTER_LOCATION_H_
