/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <brt/distributed/local_reorder.h>
#include <cuda_runtime.h>
#include <dmlc/common.h>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

namespace brt {
namespace distributed {

template <int BLOCK_THREADS, cub::BlockReduceAlgorithm ALGORITHM>
__global__ void locality_reorder(int* loads, int world_size, int* reorder_indices,
                                 int* reordered_loads) {
  int thd_id = threadIdx.x;
  __shared__ int shared_loads[BLOCK_THREADS];
  __shared__ int selected_rank_index;
  int load_length = world_size * world_size;

  typedef cub::BlockReduce<cub::KeyValuePair<int, int>, BLOCK_THREADS, ALGORITHM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int thread_current_rank_index = thd_id / world_size;
  int thread_original_rank_index = thd_id % world_size;
  int thread_global_rank_index_base = thread_current_rank_index * world_size;

  cub::KeyValuePair<int, int> thread_index_load;
  if (thd_id < load_length) {
    shared_loads[thd_id] = loads[thd_id];
    thread_index_load = cub::KeyValuePair<int, int>(thd_id, shared_loads[thd_id]);
  }

  for (int i = 0; i < world_size; i++) {
    cub::KeyValuePair<int, int> aggregate =
        BlockReduce(temp_storage).Reduce(thread_index_load, cub::ArgMax(), load_length);

    if (thd_id == 0) {
      selected_rank_index = aggregate.key;
    }
    __syncthreads();

    int new_rank_index = selected_rank_index / world_size;
    int original_rank_index = selected_rank_index % world_size;

    if (thd_id < load_length) {
      if (thread_original_rank_index == new_rank_index) {
        reordered_loads[thd_id] = shared_loads[thread_global_rank_index_base + original_rank_index];
      }
      if (thd_id >= new_rank_index * world_size && thd_id < (new_rank_index + 1) * world_size) {
        thread_index_load.value = -1;
      }
      if (thread_original_rank_index == original_rank_index) {
        thread_index_load.value = -1;
      }
    }
    if (thd_id == 0) {
      reorder_indices[new_rank_index] = original_rank_index;
    }
  }
}

template <int BLOCK_THREADS, int BLOCK_RANKS, cub::BlockReduceAlgorithm ALGORITHM>
__global__ void group_locality_reorder(int* loads, int group_size, int world_size,
                                       int* reorder_indices, int* reordered_loads) {
  const int thd_id = threadIdx.x;
  const int group_id = thd_id / group_size;
  const int group_base_thd_id = group_id * group_size;

  const int rank_load_length = world_size * world_size;
  const int total_load_length = rank_load_length * group_size;

  __shared__ int shared_loads[BLOCK_THREADS];
  __shared__ int untouched_loads[BLOCK_THREADS];
  __shared__ int shared_reorder_indices[BLOCK_RANKS];

  if (thd_id < total_load_length) {
    shared_loads[thd_id] = loads[thd_id];
    untouched_loads[thd_id] = loads[thd_id];
  }

  for (int stride = group_size / 2; stride >= 1; group_size >>= 1) {
    __syncthreads();
    if (thd_id < stride + group_base_thd_id) {
      shared_loads[thd_id] += shared_loads[thd_id + stride];
    }
  }
  if (thd_id == group_base_thd_id) {
    shared_loads[group_id] = shared_loads[group_base_thd_id];
  }

  typedef cub::BlockReduce<cub::KeyValuePair<int, int>, BLOCK_THREADS, ALGORITHM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // int thread_current_rank_index = thd_id / world_size;
  int thread_original_rank_index = thd_id % world_size;

  cub::KeyValuePair<int, int> thread_index_load;
  if (thd_id < rank_load_length) {
    thread_index_load = cub::KeyValuePair<int, int>(thd_id, shared_loads[thd_id]);
  }
  __shared__ int selected_rank_index;

  for (int i = 0; i < world_size; i++) {
    cub::KeyValuePair<int, int> aggregate =
        BlockReduce(temp_storage).Reduce(thread_index_load, cub::ArgMax(), rank_load_length);

    if (thd_id == 0) {
      selected_rank_index = aggregate.key;
    }
    __syncthreads();

    int new_rank_index = selected_rank_index / world_size;
    int original_rank_index = selected_rank_index % world_size;

    if (thd_id < rank_load_length) {
      if ((thd_id >= new_rank_index * world_size && thd_id < (new_rank_index + 1) * world_size) ||
          (thread_original_rank_index == original_rank_index)) {
        thread_index_load.value = -1;
      }
    }

    if (thd_id == 0) {
      shared_reorder_indices[new_rank_index] = original_rank_index;
    }
  }
  __syncthreads();
  if (thd_id < total_load_length) {
    int load_offset = thd_id % group_size;
    int current_rank_index = group_id % world_size;
    int base_rank_index = group_id / world_size;
    int original_rank_index = shared_reorder_indices[current_rank_index];
    int global_load_index =
        (base_rank_index * world_size + original_rank_index) * group_size + load_offset;
    reordered_loads[thd_id] = shared_loads[global_load_index];
  }
}

void LocalityReorder(int* loads, const int& world_size, int* reorder_indices, int* reordered_loads,
                     cudaStream_t stream) {
  constexpr int BLOCK_THREADS = 1024;
  CHECK_LE(world_size, 32);
  locality_reorder<BLOCK_THREADS, cub::BLOCK_REDUCE_WARP_REDUCTIONS>
      <<<1, BLOCK_THREADS, 0, stream>>>(loads, world_size, reorder_indices, reordered_loads);
}

void GroupLocalityReorder(int* loads, const int& group_size, const int& world_size,
                          int* reorder_indices, int* reordered_loads, cudaStream_t stream) {
  constexpr int BLOCK_THREADS = 1024;
  constexpr int BLOCK_RANKS = 32;
  CHECK_LE(group_size * world_size * world_size, 1024);
  group_locality_reorder<BLOCK_THREADS, BLOCK_RANKS, cub::BLOCK_REDUCE_WARP_REDUCTIONS>
      <<<1, BLOCK_THREADS, 0, stream>>>(loads, group_size, world_size, reorder_indices,
                                        reordered_loads);
}

}  // namespace distributed

}  // namespace brt
