
#include <brt/router/dispatcher/location.h>

namespace brt {
namespace router {

__global__ void calculate_location_map_capacity(
    int* __restrict__ one_hot_mask /* (num_samples, branch_num) */,
    int* __restrict__ locations /* (num_samples, branch_num) */, int* __restrict__ branch_loads,
    int* __restrict__ supported_capacities, int num_samples, int branch_num,
    int supported_capacity_num) {
  // [thread_extent] blockIdx.x = branch_num
  // [thread_extent] threadIdx.x = 1024
  constexpr int thread_num = 1024;
  __shared__ int partial_branch_mask[thread_num + 1];
  int last_sum = -1;

  for (int S = 0; S < num_samples; S += thread_num) {
    int offset = 1;
    if (S + threadIdx.x < num_samples) {
      partial_branch_mask[threadIdx.x] = one_hot_mask[threadIdx.x * branch_num + blockIdx.x];
    }
    // sum all partial_mask_per_branch[0:threadIdx.x] to partial_mask_per_branch[thread_num - 1]
    for (int d = thread_num >> 1; d > 0; d >>= 1) {
      __syncthreads();
      if (threadIdx.x < d) {
        partial_branch_mask[offset * (2 * threadIdx.x + 2) - 1] +=
            partial_branch_mask[offset * (2 * threadIdx.x + 1) - 1];
      }
      offset *= 2;
    }
    // store the sum of temp[0:threadIdx.x] to temp[thread_num] and put temp[thread_num - 1] to 0
    if (threadIdx.x == 0) {
      partial_branch_mask[thread_num] = partial_branch_mask[thread_num - 1];
      partial_branch_mask[thread_num - 1] = 0;
    }
    // reverse dispatch the sum of temp[0:threadIdx.x] to temp[threadIdx.x+1]
    for (int d = 1; d < thread_num; d *= 2) {
      offset >>= 1;
      __syncthreads();
      if (threadIdx.x < d) {
        int ai = offset * (2 * threadIdx.x + 1) - 1;
        int bi = offset * (2 * threadIdx.x + 2) - 1;
        int t = partial_branch_mask[ai];
        partial_branch_mask[ai] = partial_branch_mask[bi];
        partial_branch_mask[bi] += t;
      }
    }
    __syncthreads();
    if (S + threadIdx.x < num_samples) {
      int location = partial_branch_mask[threadIdx.x + 1] + last_sum;
      if (location == -1) location = 0;
      locations[threadIdx.x * branch_num + blockIdx.x] = location;
    }
    __syncthreads();
    last_sum += partial_branch_mask[thread_num];
    locations += thread_num * branch_num;
    one_hot_mask += thread_num * branch_num;
  }
  if (threadIdx.x == 0) {
    last_sum = last_sum + 1;
    for (int i = 0; i < supported_capacity_num; i++) {
      if (last_sum <= supported_capacities[i]) {
        branch_loads[blockIdx.x] = supported_capacities[i];
      }
    }
  }
}

void MakeLocationAndLoad(int* one_hot_mask, int* locations, int* branch_loads,
                         int* supported_capacities, int num_samples, int branch_num,
                         int supported_capacity_num, cudaStream_t stream) {
  const dim3 block_size = 1024;
  const dim3 grid_size = branch_num;
  calculate_location_map_capacity<<<grid_size, block_size, 0, stream>>>(
      one_hot_mask, locations, branch_loads, supported_capacities, num_samples, branch_num,
      supported_capacity_num);
}
}  // namespace router
}  // namespace brt
