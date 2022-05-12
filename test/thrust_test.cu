#include <brt/common/cuda_utils.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <unordered_map>
using namespace brt;

struct KernelConfig {
  // Handling JIT compilation in Multi-gpu cases
  std::vector<CUfunction> hFunc;
  std::string code, fname;
  dim3 blocks, threads;
  // branches
  std::vector<uint> grid_sizes;
  std::vector<uint> block_sizes;
  // homo branches related
  int branch_num;
  int supported_capacity_num;
  std::vector<int> supported_capacities;
  int arg_num;
  int shared_arg_num;
  int standalone_arg_num;
  std::vector<uint> shared_arg_grans;
  thrust::host_vector<thrust::device_vector<void*>> arg_ptr_array;
};

int main(int argc, char const* argv[]) {
  std::vector<KernelConfig> kernel_configs;
  kernel_configs.resize(1);
  kernel_configs[0].arg_ptr_array.resize(16, thrust::device_vector<void*>(16, nullptr));

  return 0;
}
