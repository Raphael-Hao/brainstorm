
#include <brt/router/location.h>
#include <dmlc/common.h>
namespace brt {
namespace router {

__device__ __forceinline__ void blockwise_cum_sum(int* input,
                                                  int* output_sum,
                                                  const int& cumsum_num,
                                                  int& prefix) {
  constexpr int thread_num = 1024;
  int parallel_num = gridDim.x;
  __shared__ int partial_src_mask[thread_num + 1];

  for (int S = 0; S < cumsum_num; S += thread_num) {
    partial_src_mask[threadIdx.x] = 0;
    int offset = 1;
    if (S + threadIdx.x < cumsum_num) {
      partial_src_mask[threadIdx.x] = input[threadIdx.x * parallel_num + blockIdx.x];
    }
    // sum all partial_mask_per_branch[0:threadIdx.x] to partial_mask_per_branch[thread_num - 1]
    for (int d = thread_num >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (threadIdx.x < d) {
        partial_src_mask[offset * (2 * threadIdx.x + 2) - 1] +=
            partial_src_mask[offset * (2 * threadIdx.x + 1) - 1];
      }
      offset *= 2;
    }
    // store the sum of temp[0:threadIdx.x] to temp[thread_num] and put temp[thread_num - 1] to 0
    if (threadIdx.x == 0) {
      partial_src_mask[thread_num] = partial_src_mask[thread_num - 1];
      partial_src_mask[thread_num - 1] = 0;
    }
    // reverse dispatch the sum of temp[0:threadIdx.x] to temp[threadIdx.x+1]
    for (int d = 1; d < thread_num; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (threadIdx.x < d) {
        int ai = offset * (2 * threadIdx.x + 1) - 1;
        int bi = offset * (2 * threadIdx.x + 2) - 1;
        int t = partial_src_mask[ai];
        partial_src_mask[ai] = partial_src_mask[bi];
        partial_src_mask[bi] += t;
      }
    }
    __syncthreads();
    if (S + threadIdx.x < cumsum_num) {
      int location = partial_src_mask[threadIdx.x + 1] + prefix;
      output_sum[threadIdx.x * parallel_num + blockIdx.x] = location;
    }
    __syncthreads();
    prefix += partial_src_mask[thread_num];
    output_sum += thread_num * parallel_num;
    input += thread_num * parallel_num;
  }
}

__device__ __forceinline__ void blockwise_throttle_hotmask(int* hotmask,
                                                           int* throttled_mask,
                                                           int* prefix,
                                                           int* threshold,
                                                           const int& cell_num) {
  constexpr int thread_num = 1024;
  int parallel_num = gridDim.x;
  int path_id = blockIdx.x;
  __shared__ int partial_src_mask[thread_num + 1];
  int thread_prefix = prefix[path_id];
  int thread_threshold = threshold[path_id];
  for (int S = 0; S < cell_num; S += thread_num) {
    partial_src_mask[threadIdx.x] = 0;
    int offset = 1;
    int src_mask = 0;
    if (S + threadIdx.x < cell_num) {
      partial_src_mask[threadIdx.x] = hotmask[threadIdx.x * parallel_num + path_id];
      src_mask = partial_src_mask[threadIdx.x];
    }
    // sum all partial_mask_per_branch[0:threadIdx.x] to partial_mask_per_branch[thread_num - 1]
    for (int d = thread_num >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (threadIdx.x < d) {
        partial_src_mask[offset * (2 * threadIdx.x + 2) - 1] +=
            partial_src_mask[offset * (2 * threadIdx.x + 1) - 1];
      }
      offset *= 2;
    }
    // store the sum of temp[0:threadIdx.x] to temp[thread_num] and put temp[thread_num - 1] to 0
    if (threadIdx.x == 0) {
      partial_src_mask[thread_num] = partial_src_mask[thread_num - 1];
      partial_src_mask[thread_num - 1] = 0;
    }
    // reverse dispatch the sum of temp[0:threadIdx.x] to temp[threadIdx.x+1]
    for (int d = 1; d < thread_num; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (threadIdx.x < d) {
        int ai = offset * (2 * threadIdx.x + 1) - 1;
        int bi = offset * (2 * threadIdx.x + 2) - 1;
        int t = partial_src_mask[ai];
        partial_src_mask[ai] = partial_src_mask[bi];
        partial_src_mask[bi] += t;
      }
    }
    __syncthreads();
    if (S + threadIdx.x < cell_num) {
      int location = partial_src_mask[threadIdx.x + 1] + thread_prefix;
      if (src_mask == 1 && location <= thread_threshold)
        throttled_mask[threadIdx.x * parallel_num + path_id] = 1;
    }
    __syncthreads();
    thread_prefix += partial_src_mask[thread_num];
    throttled_mask += thread_num * parallel_num;
    hotmask += thread_num * parallel_num;
    if (thread_prefix >= thread_threshold) {
      thread_prefix = thread_threshold;
      break;
    }
  }
  if (threadIdx.x == 0) {
    prefix[path_id] = thread_prefix;
  }
}

__global__ void throttle_hotmask(int* hotmask,
                                 int* throttled_mask,
                                 int* prefix,
                                 int* threshold,
                                 int cell_num) {
  blockwise_throttle_hotmask(hotmask, throttled_mask, prefix, threshold, cell_num);
}

template <bool is_tag_index>
__device__ __forceinline__ void blockwise_generate_indices(int* mask,
                                                           int* indices,
                                                           const int& cell_num,
                                                           int& prefix) {
  constexpr int thread_num = 1024;
  int parallel_num = gridDim.x;
  __shared__ int partial_src_mask[thread_num + 1];

  for (int S = 0; S < cell_num; S += thread_num) {
    partial_src_mask[threadIdx.x] = 0;
    int offset = 1;
    int src_mask = 0;
    if (S + threadIdx.x < cell_num) {
      partial_src_mask[threadIdx.x] = mask[threadIdx.x * parallel_num + blockIdx.x];
      src_mask = partial_src_mask[threadIdx.x];
    }
    // sum all partial_mask_per_branch[0:threadIdx.x] to partial_mask_per_branch[thread_num - 1]
    for (int d = thread_num >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (threadIdx.x < d) {
        partial_src_mask[offset * (2 * threadIdx.x + 2) - 1] +=
            partial_src_mask[offset * (2 * threadIdx.x + 1) - 1];
      }
      offset *= 2;
    }
    /* store the sum of temp[0:threadIdx.x] to temp[thread_num] and put temp[thread_num - 1] to 0 */
    if (threadIdx.x == 0) {
      partial_src_mask[thread_num] = partial_src_mask[thread_num - 1];
      partial_src_mask[thread_num - 1] = 0;
    }

    /* reverse dispatch the sum of temp[0:threadIdx.x] to temp[threadIdx.x+1]*/
    for (int d = 1; d < thread_num; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (threadIdx.x < d) {
        int ai = offset * (2 * threadIdx.x + 1) - 1;
        int bi = offset * (2 * threadIdx.x + 2) - 1;
        int t = partial_src_mask[ai];
        partial_src_mask[ai] = partial_src_mask[bi];
        partial_src_mask[bi] += t;
      }
    }
    __syncthreads();
    if (S + threadIdx.x < cell_num) {
      int location = partial_src_mask[threadIdx.x + 1] + prefix;
      int index = location * src_mask;
      if (index != 0) {
        if (is_tag_index) {
          indices[(index - 1) * parallel_num + blockIdx.x] = S + threadIdx.x + 1;
        } else {
          indices[threadIdx.x * parallel_num + blockIdx.x] = index;
        }
      }
    }
    __syncthreads();
    prefix += partial_src_mask[thread_num];
    mask += thread_num * parallel_num;
    if (!is_tag_index) {
      indices += thread_num * parallel_num;
    }
  }
}

/*!
 * \brief global kernel to generate indices and loads
 *
 *@tparam capacity_padding: whether to pad the load to the nearest supported capacity
 *@tparam is_tag_index : is generate tag index or seat index
 *
 * \param hot_mask [cell_num, path_num]
 * \param indices [cell_num, path_num]
 * \param loads [path_num]
 * \param cell_num: number of cells
 * \param supported_capacities: list of supported capacities
 * \param supported_capacity_num: number of supported capacities
 *
 * TODO: support padding different paths to different capacities
 */
template <bool capacity_padding, bool path_wise_padding, bool is_tag_index>
__global__ void generate_indices_and_loads(int* __restrict__ hot_mask /* [cell_num, path_num] */,
                                           int* __restrict__ indices /* [cell_num, path_num] */,
                                           int* __restrict__ loads /* [path_num] */,
                                           int cell_num,
                                           int* __restrict__ supported_capacities = nullptr,
                                           int supported_capacity_num = 0) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  int prefix = 0;
  blockwise_generate_indices<is_tag_index>(hot_mask, indices, cell_num, prefix);
  if (threadIdx.x == 0) {
    auto& real_load = prefix;
    // FIXME : path_wise_padding is padding to the supported capacity of each path unconditionally.
    //         In the future, we should not pad it if the load is zero.
    if (path_wise_padding) {
      loads[blockIdx.x] = supported_capacities[blockIdx.x];
      return;
    }
    if (real_load == 0) {
      loads[blockIdx.x] = 0;
      return;
    }
    if (capacity_padding) {
      for (int i = 0; i < supported_capacity_num; i++) {
        if (real_load <= supported_capacities[i]) {
          loads[blockIdx.x] = supported_capacities[i];
          return;
        }
      }
      loads[blockIdx.x] = supported_capacities[supported_capacity_num - 1];
    } else {
      if (supported_capacity_num == 0) {
        loads[blockIdx.x] = prefix;
        return;
      }
      for (int i = 0; i < supported_capacity_num; i++) {
        if (real_load <= supported_capacities[i]) {
          loads[blockIdx.x] = real_load;
          return;
        }
      }
      loads[blockIdx.x] = supported_capacities[supported_capacity_num - 1];
    }
  }
}

__global__ void convert_seat_to_tag_indices(
    int* __restrict__ seat_indices /* [cell_num, path_num] */,
    int* __restrict__ tag_indices /* [cell_num, path_num] */,
    int cell_num,
    int path_num) {
  // [thread_extent] blockIdx.x = path_num
  // [thread_extent] threadIdx.x = 1024
  constexpr int thread_num = 1024;
  for (int s_id = 0; s_id < cell_num; s_id += thread_num) {
    if (s_id + threadIdx.x < cell_num) {
      int cell_id = s_id + threadIdx.x;
      int dst_index_id = cell_id * path_num + blockIdx.x;
      int dst_index = seat_indices[dst_index_id];
      if (dst_index != 0) {
        int src_index_id = (dst_index - 1) * path_num + blockIdx.x;
        tag_indices[src_index_id] = cell_id + 1;
      }
    }
  }
}

__global__ void convert_tag_to_seat_indices(
    int* __restrict__ tag_indices /* [cell_num, path_num] */,
    int* __restrict__ seat_indices /* [cell_num, path_num] */,
    int* loads /* [path_num] */,
    int cell_num,
    int path_num) {
  // [thread_extent] blockIdx.x = path_num
  // [thread_extent] threadIdx.x = 1024
  constexpr int thread_num = 1024;
  int path_load = loads[blockIdx.x];
  for (int s_id = 0; s_id < path_load; s_id += thread_num) {
    if (s_id + threadIdx.x < path_load) {
      int cell_id = s_id + threadIdx.x;
      int src_index_id = cell_id * path_num + blockIdx.x;
      int src_index = tag_indices[src_index_id];
      if (src_index == 0) {
        break;
      }
      int dst_index_id = (src_index - 1) * path_num + blockIdx.x;
      seat_indices[dst_index_id] = cell_id + 1;
    }
  }
}

void ThrottleHotmask(int* hotmask,
                     int* throttled_mask,
                     int* prefix,
                     int* threshold,
                     const int& cell_num,
                     const int& path_num,
                     cudaStream_t stream) {
  const dim3 block_size = 1024;
  const dim3 grid_size = path_num;
  throttle_hotmask<<<grid_size, block_size, 0, stream>>>(hotmask, throttled_mask, prefix, threshold,
                                                         cell_num);
}

void ConvertIndexFormat(int* origin_indices,
                        int* new_indices,
                        int* loads,
                        const int& cell_num,
                        const int& path_num,
                        const bool& tag_to_seat,
                        cudaStream_t stream) {
  constexpr dim3 block_size = 1024;
  const dim3 grid_size = path_num;
  if (tag_to_seat) {
    convert_tag_to_seat_indices<<<grid_size, block_size, 0, stream>>>(origin_indices, new_indices,
                                                                      loads, cell_num, path_num);
  } else {
    convert_seat_to_tag_indices<<<grid_size, block_size, 0, stream>>>(origin_indices, new_indices,
                                                                      cell_num, path_num);
  }
}

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
                             cudaStream_t stream) {
  constexpr dim3 block_size = 1024;
  const dim3 grid_size = path_num;
  if (is_tag_index) {
    if (capacity_padding) {
      CHECK_GE(supported_capacity_num, 1);
      if (path_wise_padding) {
        generate_indices_and_loads<true, true, true><<<grid_size, block_size, 0, stream>>>(
            hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);

      } else {
        generate_indices_and_loads<true, false, true><<<grid_size, block_size, 0, stream>>>(
            hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);
      }
    } else {
      generate_indices_and_loads<false, false, true><<<grid_size, block_size, 0, stream>>>(
          hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);
    }
  } else {
    if (capacity_padding) {
      CHECK_GE(supported_capacity_num, 1);
      if (path_wise_padding) {
        generate_indices_and_loads<true, true, false><<<grid_size, block_size, 0, stream>>>(
            hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);

      } else {
        generate_indices_and_loads<true, false, false><<<grid_size, block_size, 0, stream>>>(
            hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);
      }
    } else {
      generate_indices_and_loads<false, false, false><<<grid_size, block_size, 0, stream>>>(
          hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);
    }
  }
}

}  // namespace router
}  // namespace brt
