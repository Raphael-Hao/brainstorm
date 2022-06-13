
#include <brt/router/location.h>

namespace brt {
namespace router {

__device__ __forceinline__ void blockwise_cum_sum_sub(int* input, int* output_sum,
                                                      const int& cumsum_num, int& sub_num) {
  constexpr int thread_num = 1024;
  int parallel_num = gridDim.x;
  __shared__ int partial_dst_mask[thread_num + 1];

  for (int S = 0; S < cumsum_num; S += thread_num) {
    int offset = 1;
    if (S + threadIdx.x < cumsum_num) {
      partial_dst_mask[threadIdx.x] = input[threadIdx.x * parallel_num + blockIdx.x];
    }
    // sum all partial_mask_per_branch[0:threadIdx.x] to partial_mask_per_branch[thread_num - 1]
    for (int d = thread_num >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (threadIdx.x < d) {
        partial_dst_mask[offset * (2 * threadIdx.x + 2) - 1] +=
            partial_dst_mask[offset * (2 * threadIdx.x + 1) - 1];
      }
      offset *= 2;
    }
    // store the sum of temp[0:threadIdx.x] to temp[thread_num] and put temp[thread_num - 1] to 0
    if (threadIdx.x == 0) {
      partial_dst_mask[thread_num] = partial_dst_mask[thread_num - 1];
      partial_dst_mask[thread_num - 1] = 0;
    }
    // reverse dispatch the sum of temp[0:threadIdx.x] to temp[threadIdx.x+1]
    for (int d = 1; d < thread_num; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (threadIdx.x < d) {
        int ai = offset * (2 * threadIdx.x + 1) - 1;
        int bi = offset * (2 * threadIdx.x + 2) - 1;
        int t = partial_dst_mask[ai];
        partial_dst_mask[ai] = partial_dst_mask[bi];
        partial_dst_mask[bi] += t;
      }
    }
    __syncthreads();
    if (S + threadIdx.x < cumsum_num) {
      int location = partial_dst_mask[threadIdx.x + 1] + sub_num;
      printf("%d\n", sub_num);
      output_sum[threadIdx.x * parallel_num + blockIdx.x] = location;
    }
    __syncthreads();
    sub_num += partial_dst_mask[thread_num];
    output_sum += thread_num * parallel_num;
    input += thread_num * parallel_num;
  }
}

__global__ void generate_location_and_load_map(
    int* __restrict__ hot_mask /* (sample_num, dst_num) */,
    int* __restrict__ locations /* (sample_num, dst_num) */, int* __restrict__ dst_loads,
    int* __restrict__ supported_capacities, int sample_num, int dst_num,
    int supported_capacity_num) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  int sub_num = 0;
  blockwise_cum_sum_sub(hot_mask, locations, sample_num, sub_num);
  if (threadIdx.x == 0) {
    for (int i = 0; i < supported_capacity_num; i++) {
      printf("dst: %d, real load: %d, mapped load: %d\n", blockIdx.x, sub_num,
             supported_capacities[i]);
      if (sub_num <= supported_capacities[i]) {
        dst_loads[blockIdx.x] = supported_capacities[i];
        break;
      }
    }
  }
}

__global__ void generate_branch_start_indices(int* __restrict__ dst_loads,
                                              int* __restrict__ dst_start_indices, int dst_num) {
  int sub_num = 0;
  dst_start_indices = dst_start_indices + 1;
  blockwise_cum_sum_sub(dst_loads, dst_start_indices, dst_num, sub_num);
}

__global__ void generate_route_indices(int* route_indices /* [sample_num, dst_num]  */,
                                       int* locations, int* dst_start_indices, int sample_num,
                                       int dst_num) {
  int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
  int index = 0;
  if (sample_id < sample_num) {
#pragma unroll
    for (int i = 0; i < dst_num; i++) {
      index += locations[sample_id * dst_num + i] + dst_start_indices[i];
    }
  }
  route_indices[sample_id] = index;
}

void GenerateGlobalRouteIndices(int* hot_mask, int* route_indices, int* dst_loads,
                                int* dst_start_indices, int* supported_capacities, int sample_num,
                                int dst_num, int supported_capacity_num, cudaStream_t stream) {
  {
    const dim3 block_size = 1024;
    const dim3 grid_size = dst_num;
    generate_location_and_load_map<<<grid_size, block_size, 0, stream>>>(
        hot_mask, hot_mask, dst_loads, supported_capacities, sample_num, dst_num,
        supported_capacity_num);
  }
  {
    constexpr dim3 block_size = 1024;
    constexpr dim3 grid_size = 1;
    generate_branch_start_indices<<<grid_size, block_size, 0, stream>>>(dst_loads,
                                                                        dst_start_indices, dst_num);
  }
  {
    constexpr dim3 block_size = 64;
    const dim3 grid_size = (sample_num + 63) / 64;
    generate_route_indices<<<grid_size, block_size, 0, stream>>>(
        route_indices, hot_mask, dst_start_indices, sample_num, dst_num);
  }
}

void GenerateLocalRouteIndices(int* hot_mask, int* route_indices, int* dst_loads,
                               int* supported_capacities, int sample_num, int dst_num,
                               int supported_capacity_num, cudaStream_t stream) {
  const dim3 block_size = 1024;
  const dim3 grid_size = dst_num;
  generate_location_and_load_map<<<grid_size, block_size, 0, stream>>>(
      hot_mask, route_indices, dst_loads, supported_capacities, sample_num, dst_num,
      supported_capacity_num);
}

}  // namespace router
}  // namespace brt
