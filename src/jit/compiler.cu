/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/jit/compiler.h>
#include <brt/runtime/cuda_utils.h>
#include <dmlc/common.h>

#include "./ptr_arith.cuh"
#include "./utils.h"

namespace brt {
namespace jit {

void KernelConfig::InitBranchArgStore() {
  this->standalone_arg_num = this->arg_num - this->shared_arg_num;
  CHECK_GT(this->branch_num, 0);
  CHECK_GT(this->supported_capacity_num, 0);
  CHECK_GT(this->arg_num, 0);
  CHECK_EQ(this->arg_num, this->shared_arg_num + this->standalone_arg_num);
  CUDA_CHECK(cudaMallocHost(&this->shared_arg_offset, sizeof(int) * this->branch_num));
  this->standalone_arg_hptr_array.resize(this->standalone_arg_num, nullptr);
  for (auto& host_ptr : this->standalone_arg_hptr_array) {
    CUDA_CHECK(cudaMallocHost(&host_ptr, sizeof(void*) * this->branch_num));
  }
  this->arg_dptr_array.resize(this->arg_num, nullptr);
  for (auto& device_ptr : this->arg_dptr_array) {
    CUDA_CHECK(cudaMalloc(&device_ptr, sizeof(void*) * this->branch_num));
  }
  for (auto i = 0; i < this->supported_capacity_num; i++) {
    this->capacity_index_map[this->supported_capacities[i]] = i;
  }
}

CUDACompiler::CUDACompiler() {}

CUDACompiler::~CUDACompiler() {}

CUDACompiler& CUDACompiler::GetCompiler() {
  static CUDACompiler instance;
  return instance;
}

std::string CUDACompiler::RTCompile(const char* code, const std::string& arch) {
  std::string arch_option = "--gpu-architecture=compute_" + arch;
  std::vector<const char*> param_cstrings = {
      "--restrict",        "--include-path=/usr/local/cuda/include",
      arch_option.c_str(), "--use_fast_math",
      "--std=c++14",       "--extra-device-vectorization"};
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));
  nvrtcResult nvrtc_compile_result =
      nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  if (nvrtc_compile_result != NVRTC_SUCCESS) {
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    LOG(FATAL) << "nvrtcCompileProgram failed: \n" << log;
  }

  size_t ptx_size;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return ptx;
}

CUfunction CUDACompiler::Activate(int fd, int dev) {
  auto& kernel = kernels_[fd];
  if (kernel.hFunc.size() <= static_cast<size_t>(dev)) kernel.hFunc.resize(dev + 1);

  if (kernel.hFunc[dev] == nullptr) {
    int major, minor;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::string arch = std::to_string(major) + std::to_string(minor);

    const char* source = kernel.code.data();

    std::string image;
    image = RTCompile(source, arch);
    long launch_bound =
        CaptureWithDefault(kernel.code, std::regex(R"(\s+__launch_bounds__\((\d+)\)\s+)"), 0);

    long max_registers = 65536 / launch_bound / 2;
    static CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_THREADS_PER_BLOCK,
                                     CU_JIT_MAX_REGISTERS};
    static void* values[] = {(void*)4L, (void*)launch_bound, (void*)max_registers};

    CUmodule hMod = nullptr;
    CU_CHECK(cuModuleLoadDataEx(&hMod, image.c_str(), sizeof(options) / sizeof(*options), options,
                                values));
    CHECK(nullptr != hMod);

    int func_entry = image.find(" .entry ");
    func_entry += 8;
    int func_end = image.find("(", func_entry);
    std::string func_name = image.substr(func_entry, func_end - func_entry);
    kernel.fname = func_name;
    CU_CHECK(cuModuleGetFunction(&kernel.hFunc[dev], hMod, func_name.c_str()));
    CHECK(nullptr != kernel.hFunc[dev]);
  }

  return kernel.hFunc[dev];
}

void CUDACompiler::Execute(const std::vector<const void*>& ppargs, int fd, int dev,
                           cudaStream_t stream) {
  CUfunction hfunc = Activate(fd, dev);
  auto& blocks = kernels_[fd].blocks;
  auto& threads = kernels_[fd].threads;
  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                             0, stream, (void**)ppargs.data(), nullptr));
}

void CUDACompiler::StaticExecute(const std::vector<const void*>& ppargs, int fd, int dev,
                                 cudaStream_t stream) {
  CUfunction hfunc = Activate(fd, dev);
  auto& blocks = kernels_[fd].blocks;
  auto& threads = kernels_[fd].threads;

  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                             0, stream, (void**)ppargs.data(), nullptr));
}

void CUDACompiler::HeteroExecute(const std::vector<const void*>& ppargs,
                                 const std::vector<long>& active_blocks, int fd, int dev,
                                 cudaStream_t stream) {
  CUfunction hfunc = Activate(fd, dev);
  auto& blocks = kernels_[fd].blocks;
  auto& threads = kernels_[fd].threads;
  CHECK_EQ(kernels_[fd].grid_sizes.size(), active_blocks.size());
  blocks.x = 0;
  threads.x = 0;
  for (size_t i = 0; i < active_blocks.size(); ++i) {
    if (active_blocks[i] == 0) continue;
    blocks.x += kernels_[fd].grid_sizes[i];
    threads.x = std::max(threads.x, kernels_[fd].block_sizes[i]);
  }
  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                             0, stream, (void**)ppargs.data(), nullptr));
}

void CUDACompiler::HomoExecute(const std::vector<const void*>& shared_inputs_ptr,
                               const std::vector<const void*>& standalone_inputs_ptr,
                               const std::vector<int>& branch_capacities, int fd, int dev,
                               cudaStream_t stream) {
  auto& kernel = kernels_[fd];

  // for (auto i = 0; i < kernel.shared_arg_num; i++) {
  //   printf("shared_inputs_ptr[%d] = %p\n", i, shared_inputs_ptr[i]);
  // }

  // for (auto i = 0; i < kernel.standalone_arg_num * kernel.branch_num; i++) {
  //   printf("standalone_inputs_ptr[%d] = %p\n", i, standalone_inputs_ptr[i]);
  // }

  std::vector<int> active_blocks(kernel.supported_capacity_num, 0);
  // reorder the arguments for kernel based on capacities
  auto branch_indice_with_order = SortIndice(branch_capacities);

  int real_branch_index = 0;
  // printf("runtime arg dispatch begin\n");
  for (auto branch_idx = 0; branch_idx < kernel.branch_num; branch_idx++) {
    auto& branch_idx_in_order = branch_indice_with_order[branch_idx];
    // printf("sorted branch: %d -> origin branch %d\n", branch_idx, branch_idx_in_order);
    auto& capacity = branch_capacities[branch_idx_in_order];
    // printf("capacity: %d\n", capacity);
    if (capacity == 0) continue;
    active_blocks[kernel.capacity_index_map[capacity]]++;
    // printf("active_blocks[%d] capacity updated to: %d\n", kernel.capacity_index_map[capacity],
    //  active_blocks[kernel.capacity_index_map[capacity]]);
    auto shared_arg_branch_index = std::accumulate(
        branch_capacities.begin(), branch_capacities.begin() + branch_idx_in_order, 0);
    // printf("shared_arg_branch_index: %d for branch: %d, real: %d\n", shared_arg_branch_index,
    //  branch_idx, real_branch_index);
    kernel.shared_arg_offset[real_branch_index] = shared_arg_branch_index;
    // printf("kernel.shared_arg_offset[%d] = %d\n", real_branch_index,
    //  kernel.shared_arg_offset[real_branch_index]);

    for (auto arg_idx = 0; arg_idx < (kernel.arg_num - kernel.shared_arg_num); arg_idx++) {
      kernel.standalone_arg_hptr_array[arg_idx][real_branch_index] =
          (void*)standalone_inputs_ptr[kernel.standalone_arg_num * branch_idx_in_order + arg_idx];
      // printf("branch: %d, standalone_arg_hptr_array[%d][%d] = %p\n", branch_idx, arg_idx,
      //  real_branch_index, kernel.standalone_arg_hptr_array[arg_idx][real_branch_index]);
    }
    real_branch_index++;
  }
  // print debug info
  // printf("runtime arg dispatch end\n");

  for (auto arg_idx = 0; arg_idx < kernel.arg_num; arg_idx++) {
    if (arg_idx < kernel.shared_arg_num) {
      DevicePtr2PtrArray((char**)kernel.arg_dptr_array[arg_idx], (char*)shared_inputs_ptr[arg_idx],
                         kernel.shared_arg_offset, kernel.branch_num,
                         kernel.shared_arg_grans[arg_idx], stream);
      // CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(kernel.arg_dptr_array[arg_idx],
                                 kernel.standalone_arg_hptr_array[arg_idx - kernel.shared_arg_num],
                                 real_branch_index * sizeof(void*), cudaMemcpyHostToDevice,
                                 stream));
      // CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  // for (auto cap_idx = 0; cap_idx < kernel.supported_capacity_num; cap_idx++) {
  //   printf("active_blocks[%d] = %d\n", cap_idx, active_blocks[cap_idx]);
  // }
  // geneerate culaunch config
  std::vector<const void*> pargs(kernel.arg_dptr_array.size() + active_blocks.size()),
      ppargs(kernel.arg_dptr_array.size() + active_blocks.size());
  for (int i = 0; i < (int)kernel.arg_num; ++i) {
    pargs[i] = kernel.arg_dptr_array[i];
    ppargs[i] = &pargs[i];
  }
  for (int i = (int)kernel.arg_num; i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)active_blocks[i - kernel.arg_num];
    ppargs[i] = &pargs[i];
    // ppargs[i] = (void*)&active_blocks[i - kernel.arg_num];
  }

  CUfunction hfunc = Activate(fd, dev);
  auto& blocks = kernels_[fd].blocks;
  auto& threads = kernels_[fd].threads;
  CHECK_EQ(kernels_[fd].grid_sizes.size(), active_blocks.size());
  blocks.x = 0;
  threads.x = 0;
  for (size_t i = 0; i < active_blocks.size(); ++i) {
    if (active_blocks[i] == 0) continue;
    blocks.x += kernels_[fd].grid_sizes[i] * active_blocks[i];
    threads.x = std::max(threads.x, kernels_[fd].block_sizes[i]);
  }

  // printf("blocks: %d, %d, %d\n", blocks.x, blocks.y, blocks.z);
  // printf("threads: %d, %d, %d\n", threads.x, threads.y, threads.z);

  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                             0, stream, (void**)ppargs.data(), nullptr));
}

std::pair<std::string, int> CUDACompiler::InjectSource(const std::string& headless_code) {
  int fd = kernels_.size();
  kernels_.resize(fd + 1);

  auto& kernel = kernels_[fd];
  kernel.code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + headless_code;

  std::string kernel_type_str = CaptureWithDefault(
      headless_code, std::regex(R"(\/\/\s+\[kernel_type\]\s+(\w+)\s*)"), "global");
  auto kernel_type_it = kernel_type_tb.find(kernel_type_str);
  if (kernel_type_it == kernel_type_tb.end()) {
    LOG(FATAL) << "unknown kernel type: " << kernel_type_str;
  } else {
    kernel.type = kernel_type_it->second;
  }

  switch (kernel.type) {
    case KernelType::kGlobal:
    case KernelType::kHorizFuse: {
      kernel.blocks.x = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.x\s*=\s*(\d+)\s*)"), 1);
      kernel.threads.x = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.x\s*=\s*(\d+)\s*)"), 1);
      break;
    }
    case KernelType::kHeteroFuse: {
      auto fused_kernel_grids_str = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.x\s*=\s*\[([0-9,\s]+)\])"),
          "");
      kernel.grid_sizes = ToUintVector(fused_kernel_grids_str, ',');
      auto fused_kernel_blocks_str = CaptureWithDefault(
          kernel.code,
          std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.x\s*=\s*\[([0-9,\s]+)\])"), "");
      kernel.block_sizes = ToUintVector(fused_kernel_blocks_str, ',');
      break;
    }
    case KernelType::kHomoFuse: {
      auto fused_kernel_grids_str = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.x\s*=\s*\[([0-9,\s]+)\])"),
          "");
      kernel.grid_sizes = ToUintVector(fused_kernel_grids_str, ',');
      auto fused_kernel_blocks_str = CaptureWithDefault(
          kernel.code,
          std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.x\s*=\s*\[([0-9,\s]+)\])"), "");
      kernel.block_sizes = ToUintVector(fused_kernel_blocks_str, ',');
      kernel.branch_num = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+branch_num\s*=\s*(\d+)\s*)"), 1);
      auto capacity_str = CaptureWithDefault(
          kernel.code,
          std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+supported_capacity\s*=\s*\[([0-9,\s]+)\])"),
          "");
      kernel.supported_capacities = ToIntVector(capacity_str);
      kernel.supported_capacity_num = kernel.supported_capacities.size();
      kernel.arg_num = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+arg_num\s*=\s*(\d+)\s*)"), 1);
      kernel.shared_arg_num = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+shared_arg_num\s*=\s*(\d+)\s*)"),
          1);
      auto shared_arg_grans_str = CaptureWithDefault(
          kernel.code,
          std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+shared_arg_grans\s*=\s*\[([0-9,\s]+)\])"), "");
      kernel.shared_arg_grans = ToUintVector(shared_arg_grans_str, ',');
      kernel.InitBranchArgStore();
      break;
    }
    case KernelType::kElasticHomoFuse: {
      kernel.blocks.x = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.x\s*=\s*(\d+)\s*)"), 1);
      kernel.threads.x = CaptureWithDefault(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.x\s*=\s*(\d+)\s*)"), 1);
      break;
    }
    default:
      LOG(FATAL) << "unknown kernel type";
      break;
  }
  kernel.blocks.y = CaptureWithDefault(
      kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.y\s+=\s+(\d+)\s*)"), 1);
  kernel.blocks.z = CaptureWithDefault(
      kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.z\s+=\s+(\d+)\s*)"), 1);
  kernel.threads.y = CaptureWithDefault(
      kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.y\s+=\s+(\d+)\s*)"), 1);
  kernel.threads.z = CaptureWithDefault(
      kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.z\s+=\s+(\d+)\s*)"), 1);

  return {kernel_type_str, fd};
}

}  // namespace jit
}  // namespace brt
