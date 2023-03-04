/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/router/location.h>
#include <brt/router/route.h>

#include "./torch.h"

namespace brt {
namespace backend {
namespace torch {

std::pair<::torch::Tensor, ::torch::Tensor> generate_src_indices(
    const ::torch::Tensor& hot_mask /*[cell_num x path_num]*/,
    const ::c10::optional<::torch::Tensor>& supported_capacities = {} /*[supported_capacity_num]*/,
    const bool& load_on_cpu = true) {
  CHECK_ON_CUDA(hot_mask);

  auto cell_num = hot_mask.size(0);
  auto path_num = hot_mask.size(1);

  int supported_capacity_num = 0;
  int* supported_capacities_data_ptr = nullptr;

  if (supported_capacities.has_value()) {
    CHECK_ON_CUDA(supported_capacities.value());
    supported_capacities_data_ptr = supported_capacities.value().data_ptr<int>();
    supported_capacity_num = supported_capacities.value().size(0);
  }

  ::torch::Tensor src_indices = ::torch::zeros_like(hot_mask, hot_mask.options());
  ::torch::Tensor loads = ::torch::zeros({path_num}, hot_mask.options());
  router::GenerateSrcIndices(hot_mask.data_ptr<int>(), src_indices.data_ptr<int>(),
                             loads.data_ptr<int>(), supported_capacities_data_ptr, cell_num,
                             path_num, supported_capacity_num,
                             at::cuda::getDefaultCUDAStream().stream());
  if (load_on_cpu) {
    loads = loads.cpu();
  }

  return {src_indices, loads};
}

std::pair<::torch::Tensor, ::torch::Tensor> generate_dst_indices(
    const ::torch::Tensor& hot_mask /*[cell_num x path_num]*/,
    const ::c10::optional<::torch::Tensor>& supported_capacities = {} /*[supported_capacity_num]*/,
    const bool& load_on_cpu = true) {
  CHECK_ON_CUDA(hot_mask);

  auto cell_num = hot_mask.size(0);
  auto path_num = hot_mask.size(1);

  int* supported_capacities_data_ptr = nullptr;
  int supported_capacity_num = 0;
  if (supported_capacities.has_value()) {
    CHECK_ON_CUDA(supported_capacities.value());
    supported_capacities_data_ptr = supported_capacities.value().data_ptr<int>();
    supported_capacity_num = supported_capacities.value().size(0);
  }

  ::torch::Tensor dst_indices = ::torch::zeros_like(hot_mask, hot_mask.options());
  ::torch::Tensor loads = ::torch::zeros({path_num}, hot_mask.options());
  router::GenerateDstIndices(hot_mask.data_ptr<int>(), dst_indices.data_ptr<int>(),
                             loads.data_ptr<int>(), supported_capacities_data_ptr, cell_num,
                             path_num, supported_capacity_num,
                             at::cuda::getDefaultCUDAStream().stream());
  if (load_on_cpu) {
    loads = loads.cpu();
  }
  return {dst_indices, loads};
}

std::pair<::torch::Tensor, ::torch::Tensor> generate_indices_and_loads(
    const ::torch::Tensor& hot_mask /*[cell_num, path_num]*/,
    const ::c10::optional<::torch::Tensor>& supported_capacities = {},
    const bool& capacity_padding = false,
    const bool& is_dst_index = true,
    const bool& load_on_cpu = false) {
  CHECK_ON_CUDA(hot_mask);

  auto cell_num = hot_mask.size(0);
  auto path_num = hot_mask.size(1);

  int* supported_capacities_data_ptr = nullptr;
  int supported_capacity_num = 0;

  if (supported_capacities.has_value()) {
    CHECK_ON_CUDA(supported_capacities.value());
    supported_capacities_data_ptr = supported_capacities.value().data_ptr<int>();
    supported_capacity_num = supported_capacities.value().size(0);
  }

  ::torch::Tensor indices = ::torch::zeros_like(hot_mask, hot_mask.options());
  ::torch::Tensor loads = ::torch::zeros({path_num}, hot_mask.options());
  if (is_dst_index) {
    router::GenerateIndicesAndLoads<true>(
        hot_mask.data_ptr<int>(), indices.data_ptr<int>(), loads.data_ptr<int>(), cell_num,
        path_num, supported_capacities_data_ptr, supported_capacity_num, capacity_padding,
        at::cuda::getDefaultCUDAStream().stream());
  } else {
    router::GenerateIndicesAndLoads<false>(
        hot_mask.data_ptr<int>(), indices.data_ptr<int>(), loads.data_ptr<int>(), cell_num,
        path_num, supported_capacities_data_ptr, supported_capacity_num, capacity_padding,
        at::cuda::getDefaultCUDAStream().stream());
  }
  if (load_on_cpu) {
    loads = loads.cpu();
  }
  return {indices, loads};
}

::torch::Tensor convert_index_format(
    const ::torch::Tensor& origin_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const int& target_index_fmt_id) {
  CHECK_ON_CUDA(origin_indices);
  ::torch::Tensor cuda_loads;
  if (!loads.is_cuda() && target_index_fmt_id == 1) {
    cuda_loads = loads.cuda();
  } else {
    cuda_loads = loads;
  }
  ::torch::Tensor new_indices = ::torch::zeros_like(origin_indices, origin_indices.options());
  auto cell_num = origin_indices.size(0);
  auto path_num = origin_indices.size(1);
  router::ConvertIndexFormat(origin_indices.data_ptr<int>(), new_indices.data_ptr<int>(),
                             cuda_loads.data_ptr<int>(), cell_num, path_num, target_index_fmt_id,
                             at::cuda::getDefaultCUDAStream().stream());
  return new_indices;
}

/*!
 * \brief dispatch data to different pathes according to the indices and loads
 *
 * \param in_data shape: [cell_num, sample_size]
 * \param route_indices shape: [cell_num, path_num]
 * \param loads shape: [path_num]
 * \param gates (optional) shape: [cell_num, path_num]
 * \param routing_dim available dimensions: [1, 2]
 * \param auto_pad if true, pad each path to load limit (if load_limit is not set, use max of loads)
 * \param load_limit load limit
 */
std::vector<::torch::Tensor> dispatch_with_indices_and_load(
    const ::torch::Tensor& in_data /*[cell_num x sample_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/,
    const bool& max_path_padding = false,
    const ::c10::optional<int>& cell_num_per_path = {} /* load limit*/,
    const bool& is_1d_routing = 1 /* available dimensions: [1, 2] */,
    const bool& out_fused = true,
    const bool& is_dst_index = true) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  auto data_type = in_data.dtype();

  int cell_num = in_data.size(0);
  int cell_size = in_data.numel() / cell_num;
  if (data_type == ::torch::kFloat16) {
    cell_size = cell_size / 2;
  }
  int path_num = route_indices.size(1);

  void* gates_data_ptr = nullptr;
  if (gates.has_value()) {
    CHECK_ON_CUDA(gates.value());
    CHECK_EQ(gates.value().dtype(), data_type);
    if (data_type == ::torch::kFloat16) {
      gates_data_ptr = gates.value().view({-1, 1}).repeat({1, 2}).data_ptr();
    } else if (data_type == ::torch::kFloat32) {
      gates_data_ptr = gates.value().data_ptr();
    }
  }

  ::torch::Tensor cuda_loads;

  if (!loads.is_cuda()) {
    cuda_loads = loads.cuda();
  } else {
    cuda_loads = loads;
  }

  int total_load = 0;
  int cell_num_per_path_value = 0;
  if (max_path_padding) {
    if (cell_num_per_path.has_value()) {
      cell_num_per_path_value = cell_num_per_path.value();
      total_load = cell_num_per_path_value * path_num;
    } else {
      cell_num_per_path_value = loads.max().item<int>();
      total_load = cell_num_per_path_value * path_num;
    }
  } else {
    total_load = loads.sum().item<int>();
  }

  auto out_shape = in_data.sizes().vec();
  out_shape[0] = total_load;
  auto out_data = ::torch::zeros(out_shape, in_data.options());

  if (data_type == ::torch::kFloat32) {
    router::DispatchWithIndicesAndLoads<float>(
        in_data.data_ptr(), out_data.data_ptr(), gates_data_ptr, route_indices.data_ptr<int>(),
        cuda_loads.data_ptr<int>(), cell_num, cell_size, path_num,
        at::cuda::getDefaultCUDAStream().stream(), cell_num_per_path_value, is_1d_routing,
        is_dst_index);
  } else if (data_type == ::torch::kFloat16) {
  } else {
    AT_ERROR("Unsupported data type: ", data_type);
  }

  return {out_data};
}

::torch::Tensor dispatch_with_dst_indices_1d(
    const ::torch::Tensor& in_data /*[cell_num x sample_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const bool& auto_pad = false,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);

  int cell_num = in_data.size(0);
  int sample_size = in_data.numel() / cell_num;
  int path_num = route_indices.size(1);

  int total_load = 0;
  int capacity = 0;
  if (auto_pad) {
    capacity = loads.max().item<int>();
    total_load = capacity * path_num;
  } else {
    total_load = loads.sum().item<int>();
  }
  ::torch::Tensor cuda_loads;
  if (!loads.is_cuda()) {
    cuda_loads = loads.cuda();
  } else {
    cuda_loads = loads;
  }

  void* gates_data_ptr = nullptr;
  if (gates.has_value()) {
    CHECK_ON_CUDA(gates.value());
    gates_data_ptr = gates.value().data_ptr();
  }

  auto out_shape = in_data.sizes().vec();
  out_shape[0] = total_load;
  auto out_data = ::torch::zeros(out_shape, in_data.options());
  CHECK_ON_CUDA(out_data);

  router::DispatchWithDstIndices1D<float>(in_data.data_ptr(), out_data.data_ptr(), gates_data_ptr,
                                          route_indices.data_ptr<int>(), cuda_loads.data_ptr<int>(),
                                          capacity, cell_num, sample_size, path_num,
                                          at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor padded_dispatch_with_dst_indices_1d(
    const ::torch::Tensor& in_data /*[cell_num x sample_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const int& pad_size,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);

  int cell_num = in_data.size(0);
  int sample_size = in_data.numel() / cell_num;
  int path_num = route_indices.size(1);

  int total_load = 0;
  int capacity = 0;

  capacity = pad_size;
  total_load = capacity * path_num;

  ::torch::Tensor cuda_loads;
  if (!loads.is_cuda()) {
    cuda_loads = loads.cuda();
  } else {
    cuda_loads = loads;
  }

  void* gates_data_ptr = nullptr;
  if (gates.has_value()) {
    CHECK_ON_CUDA(gates.value());
    gates_data_ptr = gates.value().data_ptr();
  }

  auto out_shape = in_data.sizes().vec();
  out_shape[0] = total_load;
  auto out_data = ::torch::zeros(out_shape, in_data.options());
  CHECK_ON_CUDA(out_data);

  router::DispatchWithDstIndices1D<float>(in_data.data_ptr(), out_data.data_ptr(), gates_data_ptr,
                                          route_indices.data_ptr<int>(), cuda_loads.data_ptr<int>(),
                                          capacity, cell_num, sample_size, path_num,
                                          at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor dispatch_with_dst_indices_2d(
    const ::torch::Tensor& in_data /*[cell_num x sample_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const bool& auto_pad = false) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  CHECK_ON_CUDA(loads);

  int cell_num = in_data.size(0);
  int sample_size = in_data.numel() / cell_num;
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
  auto out_data = ::torch::zeros(out_shape, in_data.options());
  CHECK_ON_CUDA(out_data);

  router::DispatchWithDstIndices2D<float>(
      in_data.data_ptr(), out_data.data_ptr(), route_indices.data_ptr<int>(), loads.data_ptr<int>(),
      capacity, cell_num, sample_size, path_num, at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor combine_with_src_indices(
    const ::torch::Tensor& in_data /*[load*path_num x sample_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const bool& auto_pad = false,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/,
    const ::c10::optional<::torch::Tensor>& out_data = {} /*[cell_num x sample_size]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  ::torch::Tensor cuda_loads;
  if (!loads.is_cuda()) {
    cuda_loads = loads.cuda();
  } else {
    cuda_loads = loads;
  }

  int cell_num = route_indices.size(0);
  int sample_size = in_data.numel() / in_data.size(0);
  int path_num = route_indices.size(1);

  void* gates_data_ptr = nullptr;
  if (gates.has_value()) {
    CHECK_ON_CUDA(gates.value());
    gates_data_ptr = gates.value().data_ptr();
  }

  auto out_shape = in_data.sizes().vec();

  int capacity = 0;
  if (auto_pad) {
    capacity = out_shape[0] / path_num;
  }

  ::torch::Tensor out_data_t;
  if (out_data.has_value()) {
    CHECK_ON_CUDA(out_data.value());
    out_data_t = out_data.value();
    router::ResidualCombineWithSrcIndices<float>(
        in_data.data_ptr(), out_data_t.data_ptr(), gates_data_ptr, route_indices.data_ptr<int>(),
        cuda_loads.data_ptr<int>(), capacity, cell_num, sample_size, path_num,
        at::cuda::getDefaultCUDAStream().stream());
  } else {
    out_shape[0] = cell_num;
    out_data_t = ::torch::zeros(out_shape, in_data.options());
    CHECK_ON_CUDA(out_data_t);
    router::CombineWithSrcIndices<float>(in_data.data_ptr(), out_data_t.data_ptr(), gates_data_ptr,
                                         route_indices.data_ptr<int>(), cuda_loads.data_ptr<int>(),
                                         capacity, cell_num, sample_size, path_num,
                                         at::cuda::getDefaultCUDAStream().stream());
  }
  return out_data_t;
}

}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_src_indices", &brt::backend::torch::generate_src_indices,
        "Generate a tensor for all src indices with each path's load mapped to supported capacity",
        pybind11::arg("hot_mask"), pybind11::arg("supported_capacities") = pybind11::none(),
        pybind11::arg("load_on_cpu") = true);
  m.def("generate_dst_indices", &brt::backend::torch::generate_dst_indices,
        "Generate a tensor for all dst indices with each path's load mapped to supported capacity",
        pybind11::arg("hot_mask"), pybind11::arg("supported_capacities") = pybind11::none(),
        pybind11::arg("load_on_cpu") = true);
  m.def("generate_indices_and_loads", &brt::backend::torch::generate_indices_and_loads,
        "Generate indices and loads for all paths, loads can be padded or throttled by supported "
        "capacities",
        pybind11::arg("hot_mask"), pybind11::arg("supported_capacities") = pybind11::none(),
        pybind11 ::arg("capacity_padding") = false, pybind11::arg("is_dst_index") = true,
        pybind11::arg("load_on_cpu") = false);
  m.def("convert_index_format", &brt::backend::torch::convert_index_format,
        "convert indices to the new index format");
  m.def("dispatch_with_dst_indices_1d", &brt::backend::torch::dispatch_with_dst_indices_1d,
        "Route data with local indices", pybind11::arg("in_data"), pybind11::arg("route_indices"),
        pybind11::arg("loads"), pybind11::arg("auto_pad") = false,
        pybind11::arg("gates") = pybind11::none());
  m.def("padded_dispatch_with_dst_indices_1d",
        &brt::backend::torch::padded_dispatch_with_dst_indices_1d, "Route data with local indices",
        pybind11::arg("in_data"), pybind11::arg("route_indices"), pybind11::arg("loads"),
        pybind11::arg("pad_size"), pybind11::arg("gates") = pybind11::none());
  m.def("dispatch_with_dst_indices_2d", &brt::backend::torch::dispatch_with_dst_indices_2d,
        "Route data with local indices", pybind11::arg("in_data"), pybind11::arg("route_indices"),
        pybind11::arg("loads"), pybind11::arg("auto_pad") = false);
  m.def("combine_with_src_indices", &brt::backend::torch::combine_with_src_indices,
        "Route data back with dst indices", pybind11::arg("in_data"),
        pybind11::arg("route_indices"), pybind11::arg("loads"), pybind11::arg("auto_pad") = false,
        pybind11::arg("gates") = pybind11::none(), pybind11::arg("out_data") = pybind11::none());
}