/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <brt/jit/compiler.h>
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

namespace brt {
namespace backend {

namespace torch {

static void static_invoke(const std::vector<::torch::Tensor>& ts, const std::vector<long>& args,
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
  jit::CUDACompiler::get_compiler().static_execute(ppargs, fd, dev,
                                                   at::cuda::getDefaultCUDAStream().stream());
}

static void hetero_invoke(const std::vector<::torch::Tensor>& ts,
                          const std::vector<long>& active_blocks, int fd) {
  printf("ts.size() = %d\n", (int)ts.size());
  printf("active_blocks.size() = %d\n", (int)active_blocks.size());
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
  jit::CUDACompiler::get_compiler().hetero_execute(ppargs, active_blocks, fd, dev,
                                                   at::cuda::getDefaultCUDAStream().stream());
}

static void homo_invoke(const std::vector<::torch::Tensor>& shared_inputs,
                        const std::vector<::torch::Tensor>& standalone_inputs,
                        const std::vector<long>& branch_capacities, int fd) {
  auto& compiler = jit::CUDACompiler::get_compiler();
  std::vector<const void*> shared_inputs_ptr(shared_inputs.size()),
      standalone_inputs_ptr(standalone_inputs.size());
  for (int i = 0; i < (int)shared_inputs.size(); ++i) {
    CHECK_ON_CUDA(shared_inputs[i]);
    shared_inputs_ptr[i] = shared_inputs[i].data_ptr();
  }
  for (int i = 0; i < (int)standalone_inputs.size(); ++i) {
    CHECK_ON_CUDA(standalone_inputs[i]);
    standalone_inputs_ptr[i] = standalone_inputs[i].data_ptr();
  }
  int dev = shared_inputs[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  compiler.homo_execute(shared_inputs_ptr, standalone_inputs_ptr, branch_capacities, fd, dev,
                        at::cuda::getDefaultCUDAStream().stream());
}

static void elastic_homo_invoke(const std::vector<::torch::Tensor>& ts,
                                const std::vector<long>& args, int fd) {
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
  jit::CUDACompiler::get_compiler().execute(ppargs, fd, dev,
                                            at::cuda::getDefaultCUDAStream().stream());
}

static std::pair<std::string, int> inject_source(const std::string& headless_code) {
  return jit::CUDACompiler::get_compiler().inject_source(headless_code);
}

}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("static_invoke", &brt::backend::torch::static_invoke,
        "Generic Invoke for GPU function (CUDA)");
  m.def("hetero_invoke", &brt::backend::torch::hetero_invoke,
        "Invoke for horizontal fused GPU function (CUDA) of heterogenous kernels");
  m.def("homo_invoke", &brt::backend::torch::homo_invoke, "Generic Invoke for GPU (CUDA)");
  m.def("elastic_homo_invoke", &brt::backend::torch::elastic_homo_invoke,
        "Generic Invoke for GPU (CUDA)");
  m.def("inject_source", &brt::backend::torch::inject_source, "Inject Source for GPU (CUDA)");
}