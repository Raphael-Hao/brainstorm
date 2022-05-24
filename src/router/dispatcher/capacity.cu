
#include <brt/runtime/cuda_utils.h>

__global__ void map_load_to_capacity(int* __restrict__ branch_loads,
                                     int* __restrict__ supported_capacities, int branch_num,
                                     int supported_capacity_num) {
  int thread_num = blockDim.x;
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int sh_supported_capacities[];

  // TODO map capacity read to a 2-D shm read
  for (int i = threadIdx.x; i < supported_capacity_num; i += thread_num) {
    sh_supported_capacities[i] = supported_capacities[i];
  }
  __syncthreads();
  if (global_id < branch_num) {
    for (int i = 0; i < supported_capacity_num; i++) {
      if (branch_loads[global_id] <= sh_supported_capacities[i]) {
        branch_loads[global_id] = sh_supported_capacities[i];
      }
    }
  }
}

void MapLoadToCapacity(int* branch_loads, int* supported_capacities, int branch_num,
                       int supported_capacity_num) {
  const dim3 block_size = 32;
  const dim3 grid_size = (branch_num + 31) / 32;
  map_load_to_capacity<<<grid_size, block_size, supported_capacity_num * sizeof(int)>>>(
      branch_loads, supported_capacities, branch_num, supported_capacity_num);
}

__global__ void capacity_accumulate(int* __restrict__ branch_loads, int* __restrict__ branch_num) {}