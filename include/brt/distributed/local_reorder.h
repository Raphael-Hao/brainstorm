/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#ifndef BRT_DISTRIBUTED_LOCAL_REORDER_H_
#define BRT_DISTRIBUTED_LOCAL_REORDER_H_
#include <cuda_runtime.h>

namespace brt {
namespace distributed {
void LocalityReorder(int* loads, const int& world_size, int* reorder_indices, int* reordered_loads,
                     cudaStream_t stream);
void GroupLocalityReorder(int* loads, const int& group_size, const int& world_size,
                          int* reorder_indices, int* reordered_loads, cudaStream_t stream);

}  // namespace distributed
}  // namespace brt
#endif  // BRT_DISTRIBUTED_LOCAL_REORDER_H_