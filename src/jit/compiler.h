/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once

#include <brt/common/cuda_utils.h>
#include <dmlc/common.h>

#include <fstream>

namespace brt {
namespace jit {

static std::string nvrtc_compile(const char* code, const std::string& arch) {
  std::string arch_option = "--gpu-architecture=compute_" + arch;
  std::vector<const char*> param_cstrings = {"--restrict", "--include-path=/usr/local/cuda/include",
                                             arch_option.c_str(), "--use_fast_math",
                                             "--extra-device-vectorization"};
  nvrtcProgram prog;
  std::ofstream code_file("/home/whcui/brainstorm_project/brainstorm/.cache/brt_code.cu");
  code_file << code;
  code_file.flush();
  NVRTC_CHECK(nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));
  NVRTC_CHECK(nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data()));

  size_t log_size;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));

  size_t ptx_size;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return ptx;
}

struct ModuleConfig {
  // Handling JIT compilation in Multi-gpu cases
  std::vector<CUfunction> hFunc;
  std::string code, fname;
  dim3 blocks, threads;
};

static std::vector<ModuleConfig> _gms;

inline static CUfunction jit_activate(int fd, int dev) {
  auto& gm = _gms[fd];
  if (gm.hFunc.size() <= dev) gm.hFunc.resize(dev + 1);

  if (gm.hFunc[dev] == nullptr) {
    int major, minor;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::string arch = std::to_string(major) + std::to_string(minor);

    const char *source = gm.code.data(), *pos, *tail;

    std::string image;
    image = nvrtc_compile(source, arch);

    long launch_bound;
    {
      char tag[] = " __launch_bounds__(";
      const char* pos = strstr(source, tag);
      launch_bound = pos ? std::atol(pos + sizeof(tag) - 1) : 1024L;
    }

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
    gm.fname = func_name;

    CU_CHECK(cuModuleGetFunction(&gm.hFunc[dev], hMod, func_name.c_str()));
    CHECK_NE(nullptr, gm.hFunc[dev]);
  }

  return gm.hFunc[dev];
}

static void jit_execute(const std::vector<const void*>& ppargs, int fd, int dev, const dim3& blocks,
                        const dim3& threads, cudaStream_t stream = 0) {
  CUfunction hfunc = jit_activate(fd, dev);
  CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z,
                             0, stream, (void**)ppargs.data(), nullptr));
}

static int inject_source(const std::string& headless_code) {
  int fd = _gms.size();
  _gms.resize(fd + 1);

  auto& gm = _gms[fd];
  gm.code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + headless_code;

  const char* source = headless_code.c_str();
  {
    char tag[] = "// [thread_extent] blockIdx.xdim = ";
    const char* pos = strstr(source, tag);
    gm.blocks.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] blockIdx.ydim = ";
    const char* pos = strstr(source, tag);
    gm.blocks.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] blockIdx.zdim = ";
    const char* pos = strstr(source, tag);
    gm.blocks.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] threadIdx.xdim = ";
    const char* pos = strstr(source, tag);
    gm.threads.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] threadIdx.ydim = ";
    const char* pos = strstr(source, tag);
    gm.threads.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] threadIdx.zdim = ";
    const char* pos = strstr(source, tag);
    gm.threads.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }

  return fd;
}

}  // namespace jit
}  // namespace brt