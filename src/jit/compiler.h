/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once

#include <brt/common/cuda_utils.h>
#include <dmlc/common.h>

#include <fstream>
#include <regex>
#include <unordered_map>

#include "./utils.h"

namespace brt {
namespace jit {

enum class KernelType { kGlobal, kHorizFuse, kHeteroFuse, kHomoFuse, kElasticHomoFuse };

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
  std::vector<uint> grid_sizes;
  std::vector<uint> block_sizes;
  std::vector<void*> inout_ptr_array;
};

class CUDACompiler {
 private:
  std::vector<KernelConfig> kernels_;

 public:
  CUDACompiler(/* args */);
  static CUDACompiler& get_compiler();

  std::string nvrtc_compile(const char* code, const std::string& arch) {
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

  CUfunction activate(int fd, int dev) {
    auto& kernel = kernels_[fd];
    if (kernel.hFunc.size() <= dev) kernel.hFunc.resize(dev + 1);

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
      CHECK_NE(nullptr, hMod);

      int func_entry = image.find(" .entry ");
      func_entry += 8;
      int func_end = image.find("(", func_entry);
      std::string func_name = image.substr(func_entry, func_end - func_entry);
      kernel.fname = func_name;

      CU_CHECK(cuModuleGetFunction(&kernel.hFunc[dev], hMod, func_name.c_str()));
      CHECK_NE(nullptr, kernel.hFunc[dev]);
      printf("kernel %s is activated\n", func_name.c_str());
    }

    return kernel.hFunc[dev];
  }

  void execute(const std::vector<const void*>& ppargs, int fd, int dev, cudaStream_t stream = 0) {
    CUfunction hfunc = activate(fd, dev);
    auto& blocks = kernels_[fd].blocks;
    auto& threads = kernels_[fd].threads;
    CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                               0, stream, (void**)ppargs.data(), nullptr));
  }

  void static_execute(const std::vector<const void*>& ppargs, int fd, int dev,
                      cudaStream_t stream = 0) {
    CUfunction hfunc = activate(fd, dev);
    auto& blocks = kernels_[fd].blocks;
    auto& threads = kernels_[fd].threads;
    CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                               0, stream, (void**)ppargs.data(), nullptr));
  }

  void hetero_execute(const std::vector<const void*>& ppargs,
                      const std::vector<uint>& active_blocks, int fd, int dev,
                      cudaStream_t stream = 0) {
    CUfunction hfunc = activate(fd, dev);
    auto& blocks = kernels_[fd].blocks;
    auto& threads = kernels_[fd].threads;
    CHECK_EQ(kernels_[fd].grid_sizes.size(), active_blocks.size());
    blocks.x = 0;
    threads.x = 0;
    for (auto i = 0; i < active_blocks.size(); ++i) {
      if (active_blocks[i] == 0) continue;
      blocks.x += kernels_[fd].grid_sizes[i];
      threads.x = std::max(threads.x, kernels_[fd].block_sizes[i]);
    }
    CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                               0, stream, (void**)ppargs.data(), nullptr));
  }

  void homo_execute(const std::vector<const void*>& ppargs, const std::vector<uint>& active_blocks,
                    int fd, int dev, cudaStream_t stream = 0) {
    CUfunction hfunc = activate(fd, dev);
    auto& blocks = kernels_[fd].blocks;

    auto& threads = kernels_[fd].threads;
    CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                               0, stream, (void**)ppargs.data(), nullptr));
  }

  std::pair<std::string, int> inject_source(const std::string& headless_code) {
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
            kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.xdim\s*=\s*(\d+)\s*)"),
            1);
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
        // printf("captured grid_sizes: %s\n", fused_kernel_grids_str.c_str());
        // printf("captured block_sizes: %s\n", fused_kernel_blocks_str.c_str());
        break;
      }
      case KernelType::kHomoFuse: {
        kernel.blocks.x = capture_with_default(
            kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.xdim\s*=\s*(\d+)\s*)"),
            1);
        kernel.threads.x = capture_with_default(
            kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+threadIdx.xdim\s*=\s*(\d+)\s*)"),
            1);
        break;
      }
      case KernelType::kElasticHomoFuse: {
        kernel.blocks.x = capture_with_default(
            kernel.code, std::regex(R"(\/\/\s+\[thread_extent\]\s+blockIdx.xdim\s*=\s*(\d+)\s*)"),
            1);
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
    // printf("captured blocks: %d, %d, %d\n", kernel.blocks.x, kernel.blocks.y, kernel.blocks.z);
    // printf("captured threads: %d, %d, %d\n", kernel.threads.x, kernel.threads.y,
    // kernel.threads.z); printf("gridsize: %d\n", kernel.grid_sizes.size()); printf("blocksize:
    // %d\n", kernel.block_sizes.size());
    return {kernel_type_str, fd};
  }

  ~CUDACompiler();
};

CUDACompiler::CUDACompiler(/* args */) {}

CUDACompiler::~CUDACompiler() {}

}  // namespace jit
}  // namespace brt