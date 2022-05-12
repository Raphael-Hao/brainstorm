/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include "./compiler.h"
#include "./utils.h"

namespace brt {
namespace jit {

void KernelConfig::InitBranchArgStore() {
  this->standalone_arg_num = this->arg_num - this->shared_arg_num;
  CHECK_GT(this->branch_num, 0);
  CHECK_GT(this->supported_capacity_num, 0);
  CHECK_GT(this->arg_num, 0);
  CHECK_EQ(this->arg_num, this->shared_arg_num + this->standalone_arg_num);
  this->arg_ptr_array.resize(arg_num, thrust::device_vector<void*>(branch_num));
  for (auto i = 0; i < this->supported_capacity_num; i++) {
    this->capacity_index_map[this->supported_capacities[i]] = i;
  }
}

CUDACompiler::CUDACompiler() {}

CUDACompiler::~CUDACompiler() {}

CUDACompiler& CUDACompiler::get_compiler() {
  static CUDACompiler instance;
  return instance;
}

std::string CUDACompiler::nvrtc_compile(const char* code, const std::string& arch) {
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

CUfunction CUDACompiler::activate(int fd, int dev) {
  auto& kernel = kernels_[fd];
  if (kernel.hFunc.size() <= static_cast<size_t>(dev)) kernel.hFunc.resize(dev + 1);

  if (kernel.hFunc[dev] == nullptr) {
    int major, minor;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::string arch = std::to_string(major) + std::to_string(minor);

    const char* source = kernel.code.data();

    std::string image;
    image = nvrtc_compile(source, arch);
    long launch_bound =
        capture_with_default(kernel.code, std::regex(R"(\s+__launch_bounds__\((\d+)\)\s+)"), 0);

    static CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_THREADS_PER_BLOCK};
    static void* values[] = {(void*)4L, (void*)launch_bound};

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
    printf("kernel %s is activated\n", func_name.c_str());
  }

  return kernel.hFunc[dev];
}

void CUDACompiler::execute(const std::vector<const void*>& ppargs, int fd, int dev,
                           cudaStream_t stream) {
  CUfunction hfunc = activate(fd, dev);
  auto& blocks = kernels_[fd].blocks;
  auto& threads = kernels_[fd].threads;
  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                             0, stream, (void**)ppargs.data(), nullptr));
}

void CUDACompiler::static_execute(const std::vector<const void*>& ppargs, int fd, int dev,
                                  cudaStream_t stream) {
  CUfunction hfunc = activate(fd, dev);
  auto& blocks = kernels_[fd].blocks;
  auto& threads = kernels_[fd].threads;
  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                             0, stream, (void**)ppargs.data(), nullptr));
}

void CUDACompiler::hetero_execute(const std::vector<const void*>& ppargs,
                                  const std::vector<long>& active_blocks, int fd, int dev,
                                  cudaStream_t stream) {
  CUfunction hfunc = activate(fd, dev);
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

void CUDACompiler::homo_execute(const std::vector<const void*>& shared_inputs_ptr,
                                const std::vector<const void*>& standalone_inputs_ptr,
                                const std::vector<long>& branch_capacities, int fd, int dev,
                                cudaStream_t stream) {
  auto& kernel = kernels_[fd];
  std::vector<int> active_blocks(kernel.supported_capacity_num, 0);

  // reorder the arguments for kernel based on capacities
  auto branch_indice_with_order = sort_indice(branch_capacities);
  int real_branch_index = 0;
  for (auto branch_idx = 0; branch_idx < kernel.branch_num; branch_idx++) {
    auto& branch_idx_in_order = branch_indice_with_order[branch_idx];
    auto& capacity = branch_capacities[branch_idx_in_order];
    if (capacity == 0) continue;
    active_blocks[kernel.capacity_index_map[capacity]]++;
    auto& standalone_arg_branch_idx = branch_idx_in_order;
    auto shared_arg_branch_index = std::accumulate(
        branch_capacities.begin(), branch_capacities.begin() + branch_idx_in_order, 0);
    for (auto arg_idx = 0; arg_idx < kernel.arg_num; arg_idx++) {
      if (arg_idx < kernel.shared_arg_num) {
        kernel.arg_ptr_array[arg_idx][real_branch_index] =
            (char*)shared_inputs_ptr[arg_idx] +
            shared_arg_branch_index * kernel.shared_arg_grans[arg_idx];
      } else {
        kernel.arg_ptr_array[arg_idx][real_branch_index] =
            (void*)standalone_inputs_ptr[kernel.standalone_arg_num * standalone_arg_branch_idx +
                                         arg_idx - kernel.shared_arg_num];
      }
    }
    real_branch_index++;
  }

  // geneerate culaunch config
  std::vector<const void*> pargs(kernel.arg_ptr_array.size() + active_blocks.size()),
      ppargs(kernel.arg_ptr_array.size() + active_blocks.size());
  for (int i = 0; i < (int)kernel.arg_num; ++i) {
    pargs[i] = thrust::raw_pointer_cast(kernel.arg_ptr_array[i].data());
    ppargs[i] = &pargs[i];
  }
  for (int i = (int)kernel.arg_num; i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)active_blocks[i - kernel.arg_num];
    ppargs[i] = &pargs[i];
  }

  CUfunction hfunc = activate(fd, dev);
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
  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                             0, stream, (void**)ppargs.data(), nullptr));
}

std::pair<std::string, int> CUDACompiler::inject_source(const std::string& headless_code) {
  int fd = kernels_.size();
  kernels_.resize(fd + 1);

  auto& kernel = kernels_[fd];
  kernel.code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + headless_code;

  std::string kernel_type_str = capture_with_default(
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
      kernel.blocks.x = capture_with_default(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.xdim\s*=\s*(\d+)\s*)"), 1);
      kernel.threads.x = capture_with_default(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.xdim\s*=\s*(\d+)\s*)"),
          1);
      break;
    }
    case KernelType::kHeteroFuse: {
      auto fused_kernel_grids_str = capture_with_default(
          kernel.code,
          std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.xdim\s*=\s*\[([0-9,\s]+)\])"), "");
      kernel.grid_sizes = to_uint_vector(fused_kernel_grids_str, ',');
      auto fused_kernel_blocks_str = capture_with_default(
          kernel.code,
          std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.xdim\s*=\s*\[([0-9,\s]+)\])"), "");
      kernel.block_sizes = to_uint_vector(fused_kernel_blocks_str, ',');
      break;
    }
    case KernelType::kHomoFuseV2: {
      auto fused_kernel_grids_str = capture_with_default(
          kernel.code,
          std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.xdim\s*=\s*\[([0-9,\s]+)\])"), "");
      kernel.grid_sizes = to_uint_vector(fused_kernel_grids_str, ',');
      auto fused_kernel_blocks_str = capture_with_default(
          kernel.code,
          std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.xdim\s*=\s*\[([0-9,\s]+)\])"), "");
      kernel.block_sizes = to_uint_vector(fused_kernel_blocks_str, ',');
      kernel.branch_num = capture_with_default(
          kernel.code, std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+branch_num\s*=\s*(\d+)\s*)"), 1);
      auto capacity_str = capture_with_default(
          kernel.code,
          std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+supported_capacity\s*=\s*\[([0-9,\s]+)\])"),
          "");
      kernel.supported_capacities = to_int_vector(capacity_str);
      kernel.supported_capacity_num = kernel.supported_capacities.size();
      kernel.arg_num = capture_with_default(
          kernel.code, std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+arg_num\s*=\s*(\d+)\s*)"), 1);
      kernel.shared_arg_num = capture_with_default(
          kernel.code, std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+shared_arg_num\s*=\s*(\d+)\s*)"),
          1);
      auto shared_arg_grans_str = capture_with_default(
          kernel.code,
          std::regex(R"(\/\/\s+\[homo_fuse_info\]\s+shared_arg_grans\s*=\s*\[([0-9,\s]+)\])"), "");
      kernel.shared_arg_grans = to_uint_vector(shared_arg_grans_str, ',');
      kernel.InitBranchArgStore();
      break;
    }
    case KernelType::kElasticHomoFuse: {
      kernel.blocks.x = capture_with_default(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.xdim\s*=\s*(\d+)\s*)"), 1);
      kernel.threads.x = capture_with_default(
          kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.xdim\s*=\s*(\d+)\s*)"),
          1);
      break;
    }
    default:
      LOG(FATAL) << "unknown kernel type";
      break;
  }
  kernel.blocks.y = capture_with_default(
      kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.ydim\s+=\s+(\d+)\s*)"), 1);
  kernel.blocks.z = capture_with_default(
      kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.zdim\s+=\s+(\d+)\s*)"), 1);
  kernel.threads.y = capture_with_default(
      kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.ydim\s+=\s+(\d+)\s*)"), 1);
  kernel.threads.z = capture_with_default(
      kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.zdim\s+=\s+(\d+)\s*)"), 1);
  return {kernel_type_str, fd};
}

}  // namespace jit
}  // namespace brt
