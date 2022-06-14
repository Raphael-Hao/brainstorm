/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <brt/router/location.h>
#include <brt/router/route.h>
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
std::vector<::torch::Tensor> generate_global_route_indices(
    const ::torch::Tensor& hot_mask /*[sample_num x sample_dim]*/,
    const ::torch::Tensor& supported_capacities /*[supported_capacity_num]*/) {
  CHECK_ON_CUDA(hot_mask);
  CHECK_ON_CUDA(supported_capacities);

  auto sample_num = hot_mask.size(0);
  auto dst_num = hot_mask.size(1);
  auto supported_capacity_num = supported_capacities.size(0);

  ::torch::Tensor route_indices = ::at::zeros_like(hot_mask, hot_mask.options());
  ::torch::Tensor dst_loads = ::at::zeros({dst_num}, hot_mask.options());
  ::torch::Tensor dst_start_indices = at::zeros({dst_num}, hot_mask.options());
  router::GenerateGlobalRouteIndices(
      hot_mask.data_ptr<int>(), route_indices.data_ptr<int>(), dst_loads.data_ptr<int>(),
      dst_start_indices.data_ptr<int>(), supported_capacities.data_ptr<int>(), sample_num, dst_num,
      supported_capacity_num, at::cuda::getDefaultCUDAStream().stream());
  return {route_indices, dst_loads};
}

std::vector<::torch::Tensor> generate_local_route_indices(
    const ::torch::Tensor& hot_mask /*[sample_num x sample_dim]*/,
    const ::torch::Tensor& supported_capacities /*[supported_capacity_num]*/) {
  CHECK_ON_CUDA(hot_mask);
  CHECK_ON_CUDA(supported_capacities);

  auto sample_num = hot_mask.size(0);
  auto dst_num = hot_mask.size(1);
  auto supported_capacity_num = supported_capacities.size(0);

  ::torch::Tensor route_indices = ::at::zeros_like(hot_mask, hot_mask.options());
  ::torch::Tensor dst_loads = ::at::zeros({dst_num}, hot_mask.options());
  router::GenerateLocalRouteIndices(hot_mask.data_ptr<int>(), route_indices.data_ptr<int>(),
                                    dst_loads.data_ptr<int>(), supported_capacities.data_ptr<int>(),
                                    sample_num, dst_num, supported_capacity_num,
                                    at::cuda::getDefaultCUDAStream().stream());
  return {route_indices, dst_loads};
}

::torch::Tensor route_with_local_indices(
    const ::torch::Tensor& in_data /*[sample_num x sample_dim]*/,
    const ::torch::Tensor& gates /*[sample_num x dst_num]*/,
    const ::torch::Tensor& route_indices /*[sample_num x dst_num]*/,
    const ::torch::Tensor& dst_loads /*[dst_num]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(gates);
  CHECK_ON_CUDA(route_indices);
  CHECK_ON_CUDA(dst_loads);

  int sample_num = in_data.size(0);
  int sample_dim = in_data.size(1);
  int dst_num = route_indices.size(1);
  int total_load = dst_loads.sum().item<int>();
  auto gates_data_ptr = gates.numel() > 0 ? gates.data_ptr<float>() : nullptr;

  auto out_data = ::at::zeros({total_load, sample_dim}, in_data.options());
  CHECK_ON_CUDA(out_data);

  router::RouteWithLocalIndices(in_data.data_ptr<float>(), out_data.data_ptr<float>(),
                                gates_data_ptr, route_indices.data_ptr<int>(),
                                dst_loads.data_ptr<int>(), sample_num, sample_dim, dst_num,
                                at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor route_back_with_local_indices(
    const ::torch::Tensor& in_data /*[?load*dst_num x sample_dim]*/,
    const ::torch::Tensor& gates /*[sample_num x dst_num]*/,
    const ::torch::Tensor& route_indices /*[sample_num x dst_num]*/,
    const ::torch::Tensor& dst_loads /*[dst_num]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(gates);
  CHECK_ON_CUDA(route_indices);
  CHECK_ON_CUDA(dst_loads);

  int sample_num = gates.size(0);
  int sample_dim = in_data.size(1);
  int dst_num = route_indices.size(1);
  auto gates_data_ptr = gates.numel() > 0 ? gates.data_ptr<float>() : nullptr;

  ::torch::Tensor out_data = ::at::zeros({sample_num, sample_dim}, in_data.options());
  CHECK_ON_CUDA(out_data);

  router::RouteBackWithLocalIndices(in_data.data_ptr<float>(), out_data.data_ptr<float>(),
                                    gates_data_ptr, route_indices.data_ptr<int>(),
                                    dst_loads.data_ptr<int>(), sample_num, sample_dim, dst_num,
                                    at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_global_route_indices", &brt::backend::torch::generate_global_route_indices,
        "Get global indices with each dst's load mapped to supported capacity");
  m.def("generate_local_route_indices", &brt::backend::torch::generate_local_route_indices,
        "Get local indices with each dst's load mapped to supported capacity");
  m.def("route_with_local_indices", &brt::backend::torch::route_with_local_indices,
        "Route data with local indices");
  m.def("route_back_with_local_indices", &brt::backend::torch::route_back_with_local_indices,
        "Route data with local indices");
}