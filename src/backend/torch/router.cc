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
std::vector<::torch::Tensor> generate_global_dst_indices(
    const ::torch::Tensor& hot_mask /*[sample_num x path_num]*/,
    const ::torch::Tensor& supported_capacities /*[supported_capacity_num]*/) {
  CHECK_ON_CUDA(hot_mask);
  CHECK_ON_CUDA(supported_capacities);

  auto sample_num = hot_mask.size(0);
  auto path_num = hot_mask.size(1);
  auto supported_capacity_num = supported_capacities.size(0);

  ::torch::Tensor global_dst_indices = ::at::zeros_like(hot_mask, hot_mask.options());
  ::torch::Tensor loads = ::at::zeros({path_num}, hot_mask.options());
  ::torch::Tensor base_indices = at::zeros({path_num}, hot_mask.options());
  router::GenerateGlobalDstIndices(
      hot_mask.data_ptr<int>(), global_dst_indices.data_ptr<int>(), loads.data_ptr<int>(),
      base_indices.data_ptr<int>(), supported_capacities.data_ptr<int>(), sample_num, path_num,
      supported_capacity_num, at::cuda::getDefaultCUDAStream().stream());
  return {global_dst_indices, loads};
}

std::pair<::torch::Tensor, ::torch::Tensor> generate_src_indices(
    const ::torch::Tensor& hot_mask /*[sample_num x path_num]*/,
    const ::c10::optional<::torch::Tensor>& supported_capacities =
        {} /*[supported_capacity_num]*/) {
  CHECK_ON_CUDA(hot_mask);

  // hot_mask.to(at::kInt, true);

  auto sample_num = hot_mask.size(0);
  auto path_num = hot_mask.size(1);

  int supported_capacity_num = 0;
  int* supported_capacities_data_ptr = nullptr;

  if (supported_capacities.has_value()) {
    CHECK_ON_CUDA(supported_capacities.value());
    supported_capacities_data_ptr = supported_capacities.value().data_ptr<int>();
    supported_capacity_num = supported_capacities.value().size(0);
  }

  ::torch::Tensor src_indices = ::at::zeros_like(hot_mask, hot_mask.options());
  ::torch::Tensor loads = ::at::zeros({path_num}, hot_mask.options());
  router::GenerateSrcIndices(hot_mask.data_ptr<int>(), src_indices.data_ptr<int>(),
                             loads.data_ptr<int>(), supported_capacities_data_ptr, sample_num,
                             path_num, supported_capacity_num,
                             at::cuda::getDefaultCUDAStream().stream());
  return {src_indices, loads};
}

std::pair<::torch::Tensor, ::torch::Tensor> generate_dst_indices(
    const ::torch::Tensor& hot_mask /*[sample_num x path_num]*/,
    const ::c10::optional<::torch::Tensor>& supported_capacities =
        {} /*[supported_capacity_num]*/) {
  CHECK_ON_CUDA(hot_mask);

  auto sample_num = hot_mask.size(0);
  auto path_num = hot_mask.size(1);

  int* supported_capacities_data_ptr = nullptr;
  int supported_capacity_num = 0;
  if (supported_capacities.has_value()) {
    CHECK_ON_CUDA(supported_capacities.value());
    supported_capacities_data_ptr = supported_capacities.value().data_ptr<int>();
    supported_capacity_num = supported_capacities.value().size(0);
  }

  ::torch::Tensor dst_indices = ::at::zeros_like(hot_mask, hot_mask.options());
  ::torch::Tensor loads = ::at::zeros({path_num}, hot_mask.options());
  router::GenerateDstIndices(hot_mask.data_ptr<int>(), dst_indices.data_ptr<int>(),
                             loads.data_ptr<int>(), supported_capacities_data_ptr, sample_num,
                             path_num, supported_capacity_num,
                             at::cuda::getDefaultCUDAStream().stream());
  return {dst_indices, loads};
}

::torch::Tensor convert_index_format(
    const ::torch::Tensor& origin_indices /*[sample_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/, const int& target_index_fmt_id) {
  CHECK_ON_CUDA(origin_indices);
  CHECK_ON_CUDA(loads);
  ::torch::Tensor new_indices = ::at::zeros_like(origin_indices, origin_indices.options());
  auto sample_num = origin_indices.size(0);
  auto path_num = origin_indices.size(1);
  router::ConvertIndexFormat(origin_indices.data_ptr<int>(), new_indices.data_ptr<int>(),
                             loads.data_ptr<int>(), sample_num, path_num, target_index_fmt_id,
                             at::cuda::getDefaultCUDAStream().stream());
  return new_indices;
}

::torch::Tensor dispatch_with_dst_indices_1d(
    const ::torch::Tensor& in_data /*[sample_num x sample_size]*/,
    const ::torch::Tensor& route_indices /*[sample_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/, const bool& auto_pad = false,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[sample_num x path_num]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  CHECK_ON_CUDA(loads);

  int sample_num = in_data.size(0);
  int sample_size = in_data.numel() / sample_num;
  int path_num = route_indices.size(1);

  int total_load = 0;
  int capacity = 0;
  if (auto_pad) {
    capacity = loads.max().item<int>();
    total_load = capacity * path_num;
  } else {
    total_load = loads.sum().item<int>();
  }

  float* gates_data_ptr = nullptr;
  if (gates.has_value()) {
    CHECK_ON_CUDA(gates.value());
    gates_data_ptr = gates.value().data_ptr<float>();
  }

  auto out_shape = in_data.sizes().vec();
  out_shape[0] = total_load;
  auto out_data = ::at::zeros(out_shape, in_data.options());
  CHECK_ON_CUDA(out_data);

  router::DispatchWithDstIndices1D(in_data.data_ptr<float>(), out_data.data_ptr<float>(),
                                   gates_data_ptr, route_indices.data_ptr<int>(),
                                   loads.data_ptr<int>(), capacity, sample_num, sample_size,
                                   path_num, at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor dispatch_with_dst_indices_2d(
    const ::torch::Tensor& in_data /*[sample_num x sample_size]*/,
    const ::torch::Tensor& route_indices /*[sample_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/, const bool& auto_pad = false) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  CHECK_ON_CUDA(loads);

  int sample_num = in_data.size(0);
  int sample_size = in_data.numel() / sample_num;
  int path_num = route_indices.size(1);

  int total_load = 0;
  int capacity = 0;
  if (auto_pad) {
    capacity = loads.max().item<int>();
    total_load = capacity * path_num;
  } else {
    total_load = loads.sum().item<int>();
  }

  auto out_shape = in_data.sizes().vec();
  out_shape[0] = total_load;
  auto out_data = ::at::zeros(out_shape, in_data.options());
  CHECK_ON_CUDA(out_data);

  router::DispatchWithDstIndices2D(in_data.data_ptr<float>(), out_data.data_ptr<float>(),
                                   route_indices.data_ptr<int>(), loads.data_ptr<int>(), capacity,
                                   sample_num, sample_size, path_num,
                                   at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor combine_with_src_indices(
    const ::torch::Tensor& in_data /*[?load*path_num x sample_size]*/,
    const ::torch::Tensor& route_indices /*[sample_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/, const bool& auto_pad = false,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[sample_num x path_num]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  CHECK_ON_CUDA(loads);

  int sample_num = route_indices.size(0);
  int sample_size = in_data.numel() / in_data.size(0);
  int path_num = route_indices.size(1);

  float* gates_data_ptr = nullptr;
  if (gates.has_value()) {
    CHECK_ON_CUDA(gates.value());
    gates_data_ptr = gates.value().data_ptr<float>();
  }

  auto out_shape = in_data.sizes().vec();

  int capacity = 0;
  if (auto_pad) {
    capacity = out_shape[0] / path_num;
  }

  out_shape[0] = sample_num;
  ::torch::Tensor out_data = ::at::zeros(out_shape, in_data.options());
  CHECK_ON_CUDA(out_data);

  router::CombineWithSrcIndices(in_data.data_ptr<float>(), out_data.data_ptr<float>(),
                                gates_data_ptr, route_indices.data_ptr<int>(),
                                loads.data_ptr<int>(), capacity, sample_num, sample_size, path_num,
                                at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_global_indices", &brt::backend::torch::generate_global_dst_indices,
        "Generate global indices with each dst's load mapped to supported capacity");
  m.def("generate_src_indices", &brt::backend::torch::generate_src_indices,
        "Generate a tensor for all src indices with each path's load mapped to supported capacity",
        pybind11::arg("hot_mask"), pybind11::arg("supported_capacities") = pybind11::none());
  m.def("generate_dst_indices", &brt::backend::torch::generate_dst_indices,
        "Generate a tensor for all dst indices with each path's load mapped to supported capacity",
        pybind11::arg("hot_mask"), pybind11::arg("supported_capacities") = pybind11::none());
  m.def("convert_index_format", &brt::backend::torch::convert_index_format,
        "convert indices to the new index format");
  m.def("dispatch_with_dst_indices_1d", &brt::backend::torch::dispatch_with_dst_indices_1d,
        "Route data with local indices", pybind11::arg("in_data"), pybind11::arg("route_indices"),
        pybind11::arg("loads"), pybind11::arg("auto_pad") = false,
        pybind11::arg("gates") = pybind11::none());
  m.def("dispatch_with_dst_indices_2d", &brt::backend::torch::dispatch_with_dst_indices_2d,
        "Route data with local indices", pybind11::arg("in_data"), pybind11::arg("route_indices"),
        pybind11::arg("loads"), pybind11::arg("auto_pad") = false);
  m.def("combine_with_src_indices", &brt::backend::torch::combine_with_src_indices,
        "Route data back with dst indices", pybind11::arg("in_data"),
        pybind11::arg("route_indices"), pybind11::arg("loads"), pybind11::arg("auto_pad") = false,
        pybind11::arg("gates") = pybind11::none());
}