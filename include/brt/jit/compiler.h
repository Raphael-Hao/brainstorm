/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef INCLUDE_BRT_JIT_COMPILER_H_
#define INCLUDE_BRT_JIT_COMPILER_H_

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <regex>
#include <string>
#include <unordered_map>

namespace brt {
namespace jit {

enum class KernelType { kGlobal, kHorizFuse, kHeteroFuse, kHomoFuse, kElasticHomoFuse };

static std::unordered_map<std::string, int> const DataTypeSize = {
    {"char", 1},    {"short", 2},    {"int", 4},      {"float", 4},    {"double", 8}, {"int8_t", 1},
    {"int16_t", 2}, {"int32_t", 4},  {"int64_t", 8},  {"uchar", 1},    {"ushort", 2}, {"uint", 4},
    {"uint8_t", 1}, {"uint16_t", 2}, {"uint32_t", 4}, {"uint64_t", 8},
};

static std::unordered_map<std::string, KernelType> const kernel_type_tb = {
    {"global", KernelType::kGlobal},
    {"horiz_fuse", KernelType::kHorizFuse},
    {"hetero_fuse", KernelType::kHeteroFuse},
    {"homo_fuse", KernelType::kHomoFuse},
    {"elastic_homo_fuse", KernelType::kElasticHomoFuse}};

struct KernelConfig {
  // Handling JIT compilation in Multi-gpu cases
  std::vector<CUfunction> hFunc;
  std::string code, fname;
  KernelType type;
  dim3 blocks, threads;
  // branches
  std::vector<uint> grid_sizes;
  std::vector<uint> block_sizes;
  // homo branches related
  int branch_num;
  int supported_capacity_num;
  std::vector<int> supported_capacities;
  std::unordered_map<int, int> capacity_index_map;
  int arg_num;
  int shared_arg_num;
  int standalone_arg_num;
  std::vector<uint> shared_arg_grans;

  // runtime dispatching
  int* shared_arg_offset;
  thrust::host_vector<void**> standalone_arg_hptr_array;  // allocate 2
  thrust::host_vector<void**> arg_dptr_array;
  void InitBranchArgStore();
};

class CUDACompiler {
 private:
  thrust::host_vector<KernelConfig> kernels_;

 public:
  CUDACompiler(/* args */);
  static CUDACompiler& get_compiler();

  std::string nvrtc_compile(const char* code, const std::string& arch);

  CUfunction activate(int fd, int dev);

  void execute(const std::vector<const void*>& ppargs, int fd, int dev, cudaStream_t stream = 0);

  void static_execute(const std::vector<const void*>& ppargs, int fd, int dev,
                      cudaStream_t stream = 0);

  void hetero_execute(const std::vector<const void*>& ppargs,
                      const std::vector<long>& active_blocks, int fd, int dev,
                      cudaStream_t stream = 0);

  void homo_execute(const std::vector<const void*>& shared_inputs_ptr,
                    const std::vector<const void*>& standalone_inputs_ptr,
                    const std::vector<long>& branch_capacities, int fd, int dev,
                    cudaStream_t stream = 0);

  std::pair<std::string, int> inject_source(const std::string& headless_code);

  ~CUDACompiler();
};

}  // namespace jit
}  // namespace brt
#endif  // INCLUDE_BRT_JIT_COMPILER_H_
