/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include "../compiler.h"

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

namespace brt {
namespace jit {
static void invoke(const std::vector<torch::Tensor>& ts, const std::vector<long>& args, int fd) {
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
  jit_execute(ppargs, fd, dev, _gms[fd].blocks, _gms[fd].threads,
              at::cuda::getDefaultCUDAStream().stream());
}
}  // namespace jit
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("invoke", &brt::jit::invoke, "Generic Invoke for GPU (CUDA)");
  m.def("inject_source", &brt::jit::inject_source, "Inject Source for GPU (CUDA)");
}