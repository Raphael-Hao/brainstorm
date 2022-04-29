/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once

#include <dlfcn.h>
#include <dmlc/common.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <regex>
#include <string>
#include <vector>

#include "../common/cuda_utils.h"

namespace brt {
namespace jit {

inline static std::string file_read(const char* path) {
  FILE* fp = fopen(path, "rb");
  CHECK_EQ(true, fp != nullptr);
  fseek(fp, 0, SEEK_END);
  size_t code_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  std::string code;
  code.resize(code_size);
  CHECK_EQ(code_size, fread((void*)code.data(), 1, code_size, fp));
  fclose(fp);
  return code;
}

inline static void file_write(const char* path, const std::string& code) {
  FILE* fp = fopen(path, "wb");
  CHECK_EQ(true, fp != nullptr);
  CHECK_EQ(code.size(), fwrite((void*)code.data(), 1, code.size(), fp));
  fclose(fp);
}

inline static std::string get_cache_path() {
  char* home_path;
  struct stat st = {0};
  if ((home_path = getenv("HOME")) == NULL) {
    home_path = getpwuid(getuid())->pw_dir;
  }
  std::string cache_path(home_path);
  cache_path += "/.cache/";
  if (stat(cache_path.c_str(), &st) == -1) {
    mkdir(cache_path.c_str(), 0755);
  }
  cache_path += "brt/";
  if (stat(cache_path.c_str(), &st) == -1) {
    mkdir(cache_path.c_str(), 0755);
  }
  cache_path += "kernels/";
  if (stat(cache_path.c_str(), &st) == -1) {
    mkdir(cache_path.c_str(), 0755);
  }

  return cache_path;
}

static std::string nvcc_compile(const char* code, const std::string& arch) {
  char code_path[] = "/tmp/torch-tutel-XXXXXX.cu";
  CHECK_NE(-1, mkstemps(code_path, 3));

  file_write(code_path, code);
  std::string fatbin_path = code_path + std::string(".fatbin");
  const char* entry = "/usr/local/cuda/bin/nvcc";

  if (access(entry, F_OK) != 0) return "";
  pid_t pid = fork();
  if (pid == 0) {
    CHECK_EQ(
        -1, execl(entry, entry, code_path, "-o", fatbin_path.c_str(), "--fatbin", "-O4", "-gencode",
                  ("arch=compute_" + arch + ",code=sm_" + arch).c_str(), (char*)NULL));
    exit(1);
  } else {
    wait(NULL);
  }
  auto image = file_read(fatbin_path.data());
  unlink(fatbin_path.data());
  unlink(code_path);
  return image;
}

static std::string nvrtc_compile(const char* code, const std::string& arch) {
  std::string arch_option = "--gpu-architecture=compute_" + arch;
  std::vector<const char*> param_cstrings = {"--restrict", "--include-path=/usr/local/cuda/include",
                                             arch_option.c_str(), "--use_fast_math",
                                             "--extra-device-vectorization"};
  nvrtcProgram prog;

  CHECK_EQ(0, nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));
  nvrtcResult res = nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  size_t log_size;
  CHECK_EQ(0, nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  CHECK_EQ(0, nvrtcGetProgramLog(prog, &log[0]));
  if (0 != res) {
    static bool once_flag = false;
    if (!once_flag) {
      once_flag = true;
      LOG(WARNING)
          << log
          << " Failed to use NVRTC for JIT compilation in this Pytorch version, try another "
             "approach using CUDA compiler.. (To always disable NVRTC, please: export USE_NVRTC=0)";
    }
    CHECK_EQ(0, nvrtcDestroyProgram(&prog));
    return "";
  }

  size_t ptx_size;
  CHECK_EQ(0, nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  CHECK_EQ(0, nvrtcGetPTX(prog, &ptx[0]));
  CHECK_EQ(0, nvrtcDestroyProgram(&prog));
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
#if !defined(__HIP_PLATFORM_HCC__)
    int major, minor;
    CHECK_EQ(0, cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CHECK_EQ(0, cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    std::string arch = std::to_string(major) + std::to_string(minor);
#else
    hipDeviceProp_t prop;
    CHECK_EQ(0, hipGetDeviceProperties(&prop, dev));
    std::string arch = prop.gcnArchName;
#endif
    const char *source = gm.code.data(), *pos, *tail;

    int use_nvrtc = getenv("USE_NVRTC") ? std::atoi(getenv("USE_NVRTC")) : 0;
    std::string image;
    if (use_nvrtc || (image = nvcc_compile(source, arch)) == "") {
      image = nvrtc_compile(source, arch);
    }

    long launch_bound;
    {
      char tag[] = " __launch_bounds__(";
      const char* pos = strstr(source, tag);
      launch_bound = pos ? std::atol(pos + sizeof(tag) - 1) : 1024L;
    }

    static CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_THREADS_PER_BLOCK};
    static void* values[] = {(void*)4L, (void*)launch_bound};

    CUmodule hMod = nullptr;
    CHECK_EQ(0, cuModuleLoadDataEx(&hMod, image.c_str(), sizeof(options) / sizeof(*options),
                                   options, values));
    CHECK_NE(nullptr, hMod);

    CHECK_NE(nullptr, (pos = strstr(source, " void ")));
    pos += 6;
    CHECK_NE(nullptr, (tail = strchr(pos, '(')));

    std::string fname = std::string(pos, tail - pos);
    gm.fname = fname;
    CHECK_EQ(0, cuModuleGetFunction(&gm.hFunc[dev], hMod, fname.c_str()));
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
#if !defined(__HIP_PLATFORM_HCC__)
  gm.code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + headless_code;
#else
  gm.code = "#include <hip/hip_runtime.h>\n" + headless_code;
#endif

  const char* source = headless_code.c_str();
  {
    char tag[] = "// [thread_extent] blockIdx.x = ";
    const char* pos = strstr(source, tag);
    gm.blocks.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] blockIdx.y = ";
    const char* pos = strstr(source, tag);
    gm.blocks.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] blockIdx.z = ";
    const char* pos = strstr(source, tag);
    gm.blocks.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] threadIdx.x = ";
    const char* pos = strstr(source, tag);
    gm.threads.x = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] threadIdx.y = ";
    const char* pos = strstr(source, tag);
    gm.threads.y = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }
  {
    char tag[] = "// [thread_extent] threadIdx.z = ";
    const char* pos = strstr(source, tag);
    gm.threads.z = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
  }

  return fd;
}

static void invoke(const std::vector<torch::Tensor>& ts, const std::vector<long>& args, int fd) {
  std::vector<const void*> pargs(ts.size() + args.size()), ppargs(ts.size() + args.size());
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr(), ppargs[i] = &pargs[i];
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)args[i - ts.size()], ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  jit_execute(ppargs, fd, dev, _gms[fd].blocks, _gms[fd].threads,
              at::cuda::getDefaultCUDAStream().stream());
}

}  // namespace jit
}  // namespace brt