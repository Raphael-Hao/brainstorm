/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <brt/router/dispatcher/location.h>
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
static void generate_indices_with_load_map(const ::torch::Tensor& one_hot_mask,
                                           const ::torch::Tensor& route_indices,
                                           const ::torch::Tensor& branch_loads,
                                           const ::torch::Tensor& branch_start_indices,
                                           const ::torch::Tensor& supported_capacities,
                                           const int& sample_num, const int& branch_num,
                                           const int& supported_capacity_num) {
  CHECK_ON_CUDA(one_hot_mask);
  CHECK_ON_CUDA(route_indices);
  CHECK_ON_CUDA(branch_loads);
  CHECK_ON_CUDA(supported_capacities);
  router::GenerateIndicesWithLoadMap(
      one_hot_mask.data_ptr<int>(), route_indices.data_ptr<int>(), branch_loads.data_ptr<int>(),
      branch_start_indices.data_ptr<int>(), supported_capacities.data_ptr<int>(), sample_num,
      branch_num, supported_capacity_num, at::cuda::getDefaultCUDAStream().stream());
}

}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_indices_with_load_map", &brt::backend::torch::generate_indices_with_load_map,
        "Get indices and map load to supported capacity");
}