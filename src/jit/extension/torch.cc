/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_ON_CPU
#undef CHECK_ON_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK_EQ(x, y) TORCH_INTERNAL_ASSERT((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) TORCH_INTERNAL_ASSERT((x) != (y), "CHECK_NE fails.")
#define CHECK_ON_CPU(x) TORCH_INTERNAL_ASSERT(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_ON_CUDA(x) TORCH_INTERNAL_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_INTERNAL_ASSERT(x.is_contiguous(), #x " must be contiguous")

#include "../compiler.h"

namespace brt {
namespace jit {

CUDACompiler& CUDACompiler::get_compiler() {
  static CUDACompiler instance;
  return instance;
}

static void static_invoke(const std::vector<torch::Tensor>& ts, const std::vector<long>& args,
                          int fd) {
  std::vector<const void*> pargs(ts.size() + args.size()), ppargs(ts.size() + args.size());
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_ON_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr();
    ppargs[i] = &pargs[i];
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)args[i - ts.size()];
    ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  CUDACompiler::get_compiler().static_execute(ppargs, fd, dev,
                                              at::cuda::getDefaultCUDAStream().stream());
}

static void hetero_invoke(const std::vector<torch::Tensor>& ts,
                          const std::vector<long>& active_blocks, int fd) {
  std::vector<const void*> pargs(ts.size() + active_blocks.size()),
      ppargs(ts.size() + active_blocks.size());

  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_ON_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr();
    ppargs[i] = &pargs[i];
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)active_blocks[i - ts.size()];
    ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  CUDACompiler::get_compiler().hetero_execute(
      ppargs, std::vector<uint>(active_blocks.begin(), active_blocks.end()), fd, dev,
      at::cuda::getDefaultCUDAStream().stream());
}

static void homo_invoke(const std::vector<torch::Tensor>& inout_ts,
                        const std::vector<torch::Tensor>& w_ts, const std::vector<long>& args,
                        int fd) {
  std::vector<const void*> pargs(inout_ts.size() + args.size()),
      ppargs(inout_ts.size() + args.size());
  for (int i = 0; i < (int)inout_ts.size(); ++i) {
    CHECK_ON_CUDA(inout_ts[i]);
    pargs[i] = inout_ts[i].data_ptr();
    ppargs[i] = &pargs[i];
  }
  for (int i = (int)inout_ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)args[i - inout_ts.size()];
    ppargs[i] = &pargs[i];
  }

  int dev = inout_ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  CUDACompiler::get_compiler().homo_execute(ppargs, std::vector<uint>(args.begin(), args.end()), fd,
                                            dev, at::cuda::getDefaultCUDAStream().stream());
}

static void elastic_homo_invoke(const std::vector<torch::Tensor>& ts, const std::vector<long>& args,
                                int fd) {
  std::vector<const void*> pargs(ts.size() + args.size()), ppargs(ts.size() + args.size());
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_ON_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr();
    ppargs[i] = &pargs[i];
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)args[i - ts.size()];
    ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  CUDACompiler::get_compiler().execute(ppargs, fd, dev, at::cuda::getDefaultCUDAStream().stream());
}

static std::pair<std::string, int> inject_source(const std::string& headless_code) {
  return CUDACompiler::get_compiler().inject_source(headless_code);
}

}  // namespace jit
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("static_invoke", &brt::jit::static_invoke, "Generic Invoke for GPU function (CUDA)");
  m.def("hetero_invoke", &brt::jit::hetero_invoke, "Invoke for horizontal fused GPU function (CUDA) of heterogenous kernels");
  m.def("homo_invoke", &brt::jit::homo_invoke, "Generic Invoke for GPU (CUDA)");
  m.def("elastic_homo_invoke", &brt::jit::elastic_homo_invoke, "Generic Invoke for GPU (CUDA)");
  m.def("inject_source", &brt::jit::inject_source, "Inject Source for GPU (CUDA)");
}