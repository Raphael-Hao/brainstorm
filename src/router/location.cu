
#include <brt/router/location.h>
#include <dmlc/common.h>
namespace brt {
namespace router {

__device__ __forceinline__ void blockwise_generate_src_indices(int* mask,
                                                               int* output_sum,
                                                               const int& cumsum_num,
                                                               int& prefix) {
  constexpr int thread_num = 1024;
  int parallel_num = gridDim.x;
  __shared__ int partial_src_mask[thread_num + 1];

  for (int S = 0; S < cumsum_num; S += thread_num) {
    partial_src_mask[threadIdx.x] = 0;
    int offset = 1;
    int src_mask = 0;
    if (S + threadIdx.x < cumsum_num) {
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
    if (S + threadIdx.x < cumsum_num) {
      int location = partial_src_mask[threadIdx.x + 1] + prefix;
      int index = location * src_mask;
      if (index != 0) {
        output_sum[(index - 1) * parallel_num + blockIdx.x] = S + threadIdx.x + 1;
        // output_sum[(index - 1) + blockIdx.x * cumsum_num] = S + threadIdx.x;
      }
    }
    __syncthreads();
    prefix += partial_src_mask[thread_num];
    mask += thread_num * parallel_num;
  }
}

__device__ __forceinline__ void blockwise_mask_cum_sum(int* mask,
                                                       int* output_sum,
                                                       const int& cumsum_num,
                                                       int& prefix) {
  constexpr int thread_num = 1024;
  int parallel_num = gridDim.x;
  __shared__ int partial_src_mask[thread_num + 1];

  for (int S = 0; S < cumsum_num; S += thread_num) {
    partial_src_mask[threadIdx.x] = 0;
    int offset = 1;
    int src_mask = 0;
    if (S + threadIdx.x < cumsum_num) {
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
      output_sum[threadIdx.x * parallel_num + blockIdx.x] = location * src_mask;
    }
    __syncthreads();
    prefix += partial_src_mask[thread_num];
    output_sum += thread_num * parallel_num;
    mask += thread_num * parallel_num;
  }
}

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
template <bool capacity_padding>
__global__ void generate_src_indices_and_loads(
    int* __restrict__ hot_mask /* [cell_num, path_num] */,
    int* __restrict__ src_indices /* [cell_num, path_num] */,
    int* __restrict__ loads /* [path_num] */,
    int cell_num,
    int* __restrict__ supported_capacities = nullptr,
    int supported_capacity_num = 0) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  int prefix = 0;
  blockwise_generate_src_indices(hot_mask, src_indices, cell_num, prefix);
  if (threadIdx.x == 0) {
    auto& real_load = prefix;
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

__global__ void generate_src_indices_and_load(
    int* __restrict__ hot_mask /* [cell_num, path_num] */,
    int* __restrict__ src_indices /* [cell_num, path_num] */,
    int* __restrict__ loads /* [path_num] */,
    int cell_num) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  int prefix = 0;
  blockwise_generate_src_indices(hot_mask, src_indices, cell_num, prefix);
  if (threadIdx.x == 0) {
    loads[blockIdx.x] = prefix;
  }
}

__global__ void generate_src_indices_and_load_map(
    int* __restrict__ hot_mask /* [cell_num, path_num] */,
    int* __restrict__ src_indices /* [cell_num, path_num] */,
    int* __restrict__ loads /* [path_num] */,
    int* __restrict__ supported_capacities,
    int cell_num,
    int supported_capacity_num) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  int prefix = 0;
  blockwise_generate_src_indices(hot_mask, src_indices, cell_num, prefix);
  if (threadIdx.x == 0) {
    auto& real_load = prefix;
    for (int i = 0; i < supported_capacity_num; i++) {
      if (real_load <= supported_capacities[i]) {
        loads[blockIdx.x] = supported_capacities[i];
        break;
      }
    }
  }
}

template <bool capacity_padding>
__global__ void generate_dst_indices_and_loads(
    int* __restrict__ hot_mask /* [cell_num, path_num] */,
    int* __restrict__ dst_indices /* [cell_num, path_num] */,
    int* __restrict__ loads /* [path_num] */,
    int cell_num,
    int* __restrict__ supported_capacities = nullptr /* [supported_capacity_num]*/,
    int supported_capacity_num = 0) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  int prefix = 0;
  blockwise_mask_cum_sum(hot_mask, dst_indices, cell_num, prefix);
  if (threadIdx.x == 0) {
    auto& real_load = prefix;
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

__global__ void generate_dst_indices_and_load(
    int* __restrict__ hot_mask /* [cell_num, path_num] */,
    int* __restrict__ dst_indices /* [cell_num, path_num] */,
    int* __restrict__ loads /* [path_num] */,
    int cell_num) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  int prefix = 0;
  blockwise_mask_cum_sum(hot_mask, dst_indices, cell_num, prefix);
  if (threadIdx.x == 0) {
    loads[blockIdx.x] = prefix;
  }
}

__global__ void generate_dst_indices_and_load_map(
    int* __restrict__ hot_mask /* [cell_num, path_num] */,
    int* __restrict__ dst_indices /* [cell_num, path_num] */,
    int* __restrict__ loads /* [path_num] */,
    int* __restrict__ supported_capacities /* [supported_capacity_num]*/,
    int cell_num,
    int supported_capacity_num) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  int prefix = 0;
  blockwise_mask_cum_sum(hot_mask, dst_indices, cell_num, prefix);
  if (threadIdx.x == 0) {
    auto& real_load = prefix;
    for (int i = 0; i < supported_capacity_num; i++) {
      if (real_load <= supported_capacities[i]) {
        loads[blockIdx.x] = supported_capacities[i];
        break;
      }
      loads[blockIdx.x] = supported_capacities[supported_capacity_num - 1];
    }
  }
}

__global__ void convert_dst_to_src_indices(int* __restrict__ dst_indices /* [cell_num, path_num] */,
                                           int* __restrict__ src_indices /* [cell_num, path_num] */,
                                           int cell_num,
                                           int path_num) {
  // [thread_extent] blockIdx.x = path_num
  // [thread_extent] threadIdx.x = 1024
  constexpr int thread_num = 1024;
  for (int s_id = 0; s_id < cell_num; s_id += thread_num) {
    if (s_id + threadIdx.x < cell_num) {
      int sample_id = s_id + threadIdx.x;
      int dst_index_id = sample_id * path_num + blockIdx.x;
      int dst_index = dst_indices[dst_index_id];
      if (dst_index != 0 && dst_index) {
        int src_index_id = (dst_index - 1) * path_num + blockIdx.x;
        src_indices[src_index_id] = sample_id;
      }
    }
  }
}

__global__ void convert_src_to_dst_indices(int* __restrict__ src_indices /* [cell_num, path_num] */,
                                           int* __restrict__ dst_indices /* [cell_num, path_num] */,
                                           int* loads /* [path_num] */,
                                           int cell_num,
                                           int path_num) {
  // [thread_extent] blockIdx.x = path_num
  // [thread_extent] threadIdx.x = 1024
  constexpr int thread_num = 1024;
  __shared__ int load;
  load = loads[blockIdx.x];
  for (int s_id = 0; s_id < load; s_id += thread_num) {
    if (s_id + threadIdx.x < load) {
      int sample_id = s_id + threadIdx.x;
      int src_index_id = sample_id * path_num + blockIdx.x;
      int src_index = src_indices[src_index_id];
      int dst_index_id = src_index * path_num + blockIdx.x;
      dst_indices[dst_index_id] = sample_id + 1;
    }
  }
}

void ConvertIndexFormat(int* origin_indices,
                        int* new_indices,
                        int* loads,
                        const int& cell_num,
                        const int& path_num,
                        const int& target_index_fmt_id,
                        cudaStream_t stream) {
  constexpr dim3 block_size = 1024;
  const dim3 grid_size = path_num;
  if (target_index_fmt_id == 0) {  // dst_indices -> src_indices
    convert_dst_to_src_indices<<<grid_size, block_size, 0, stream>>>(origin_indices, new_indices,
                                                                     cell_num, path_num);
  } else if (target_index_fmt_id == 1) {  // src_indices -> dst_indices
    convert_src_to_dst_indices<<<grid_size, block_size, 0, stream>>>(origin_indices, new_indices,
                                                                     loads, cell_num, path_num);
  }
}

void GenerateSrcIndices(int* hot_mask /*[cell_num x path_num]*/,
                        int* src_indices /*[cell_num x path_num]*/,
                        int* loads /*[path_num]*/,
                        int* supported_capacities /*[supported_capacity_num]*/,
                        const int& cell_num,
                        const int& path_num,
                        const int& supported_capacity_num,
                        cudaStream_t stream) {
  const dim3 block_size = 1024;
  const dim3 grid_size = path_num;
  if (supported_capacity_num == 0) {
    generate_src_indices_and_load<<<grid_size, block_size, 0, stream>>>(hot_mask, src_indices,
                                                                        loads, cell_num);
  } else {
    generate_src_indices_and_load_map<<<grid_size, block_size, 0, stream>>>(
        hot_mask, src_indices, loads, supported_capacities, cell_num, supported_capacity_num);
  }
}

void GenerateDstIndices(int* hot_mask /*[cell_num x path_num]*/,
                        int* dst_indices /*[cell_num x path_num]*/,
                        int* loads /*[path_num]*/,
                        int* supported_capacities /*[supported_capacity_num]*/,
                        const int& cell_num,
                        const int& path_num,
                        const int& supported_capacity_num,
                        cudaStream_t stream) {
  const dim3 block_size = 1024;
  const dim3 grid_size = path_num;
  if (supported_capacity_num == 0) {
    generate_dst_indices_and_load<<<grid_size, block_size, 0, stream>>>(hot_mask, dst_indices,
                                                                        loads, cell_num);
  } else {
    CHECK_GE(supported_capacity_num, 1);
    generate_dst_indices_and_load_map<<<grid_size, block_size, 0, stream>>>(
        hot_mask, dst_indices, loads, supported_capacities, cell_num, supported_capacity_num);
  }
}

template <bool is_dst_index>
void GenerateIndicesAndLoads(int* hot_mask /*[cell_num x path_num]*/,
                             int* indices /*[cell_num x path_num]*/,
                             int* loads /*[path_num]*/,
                             const int& cell_num,
                             const int& path_num,
                             int* supported_capacities /*[supported_capacity_num]*/,
                             const int& supported_capacity_num,
                             const bool& capacity_padding,
                             cudaStream_t stream) {
  const dim3 block_size = 1024;
  const dim3 grid_size = path_num;
  if (is_dst_index) {
    if (capacity_padding) {
      CHECK_GE(supported_capacity_num, 1);
      generate_dst_indices_and_loads<true><<<grid_size, block_size, 0, stream>>>(
          hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);
    } else {
      generate_dst_indices_and_loads<false><<<grid_size, block_size, 0, stream>>>(
          hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);
    }
  } else {
    if (capacity_padding) {
      CHECK_GE(supported_capacity_num, 1);
      generate_src_indices_and_loads<true><<<grid_size, block_size, 0, stream>>>(
          hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);
    } else {
      generate_src_indices_and_loads<false><<<grid_size, block_size, 0, stream>>>(
          hot_mask, indices, loads, cell_num, supported_capacities, supported_capacity_num);
    }
  }
}

template void GenerateIndicesAndLoads<true>(int* hot_mask /*[cell_num x path_num]*/,
                                            int* indices /*[cell_num x path_num]*/,
                                            int* loads /*[path_num]*/,
                                            const int& cell_num,
                                            const int& path_num,
                                            int* supported_capacities /*[supported_capacity_num]*/,
                                            const int& supported_capacity_num,
                                            const bool& capacity_padding,
                                            cudaStream_t stream);
template void GenerateIndicesAndLoads<false>(int* hot_mask /*[cell_num x path_num]*/,
                                             int* indices /*[cell_num x path_num]*/,
                                             int* loads /*[path_num]*/,
                                             const int& cell_num,
                                             const int& path_num,
                                             int* supported_capacities /*[supported_capacity_num]*/,
                                             const int& supported_capacity_num,
                                             const bool& capacity_padding,
                                             cudaStream_t stream);
}  // namespace router
}  // namespace brt
