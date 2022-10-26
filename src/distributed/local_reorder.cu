/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <brt/distributed/local_reorder.h>
#include <dmlc/common.h>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

namespace brt {
namespace distributed {

template <int BLOCK_THREADS, cub::BlockReduceAlgorithm ALGORITHM>
__global__ void locality_reorder(int* loads, int paths, int* reorder_indices,
                                 int* reordered_loads) {
  int thd_id = threadIdx.x;
  __shared__ int shared_loads[BLOCK_THREADS];
  __shared__ int selected_path_index;
  int load_length = paths * paths;

  typedef cub::BlockReduce<cub::KeyValuePair<int, int>, BLOCK_THREADS, ALGORITHM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int thread_current_path_index = thd_id / paths;
  int thread_original_path_index = thd_id % paths;
  int thread_global_path_index_base = thread_current_path_index * paths;

  cub::KeyValuePair<int, int> thread_index_load;
  if (thd_id < load_length) {
    shared_loads[thd_id] = loads[thd_id];
    thread_index_load = cub::KeyValuePair<int, int>(thd_id, shared_loads[thd_id]);
  }

  for (int i = 0; i < paths; i++) {
    cub::KeyValuePair<int, int> aggregate =
        BlockReduce(temp_storage).Reduce(thread_index_load, cub::ArgMax(), load_length);

    if (thd_id == 0) {
      selected_path_index = aggregate.key;
    }
    __syncthreads();

    int new_path_index = selected_path_index / paths;
    int original_path_index = selected_path_index % paths;

    if (thd_id < load_length) {
      if (thread_original_path_index == new_path_index) {
        reordered_loads[thd_id] = shared_loads[thread_global_path_index_base + original_path_index];
      }
      if (thd_id >= new_path_index * paths && thd_id < (new_path_index + 1) * paths) {
        thread_index_load.value = -1;
      }
      if (thread_original_path_index == original_path_index) {
        thread_index_load.value = -1;
      }
    }
    if (thd_id == 0) {
      reorder_indices[new_path_index] = original_path_index;
    }
  }
}

void LocalityReorder(int* loads, const int& world_size, int* reorder_indices, int* reordered_loads,
                     cudaStream_t stream) {
  constexpr int BLOCK_THREADS = 1024;
  CHECK_LE(world_size, 32);
  locality_reorder<BLOCK_THREADS, cub::BLOCK_REDUCE_WARP_REDUCTIONS>
      <<<1, BLOCK_THREADS, 0, stream>>>(loads, world_size, reorder_indices, reordered_loads);
}
}  // namespace distributed

}  // namespace brt
