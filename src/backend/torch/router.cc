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
  router::GenerateIndicesAndLoads(
      hot_mask.data_ptr<int>(), indices.data_ptr<int>(), loads.data_ptr<int>(), cell_num, path_num,
      supported_capacities_data_ptr, supported_capacity_num, capacity_padding, is_dst_index,
      at::cuda::getDefaultCUDAStream().stream());
  if (load_on_cpu) {
    loads = loads.cpu();
  }
  return {indices, loads};
}

::torch::Tensor convert_index_format(
    const ::torch::Tensor& origin_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const bool& dst_to_src) {
  CHECK_ON_CUDA(origin_indices);
  ::torch::Tensor cuda_loads;
  if (!loads.is_cuda() && !dst_to_src) {
    cuda_loads = loads.cuda();
  } else {
    cuda_loads = loads;
  }
  ::torch::Tensor new_indices = ::torch::zeros_like(origin_indices, origin_indices.options());
  auto cell_num = origin_indices.size(0);
  auto path_num = origin_indices.size(1);
  router::ConvertIndexFormat(origin_indices.data_ptr<int>(), new_indices.data_ptr<int>(),
                             cuda_loads.data_ptr<int>(), cell_num, path_num, dst_to_src,
                             at::cuda::getDefaultCUDAStream().stream());
  return new_indices;
}

/*!
 * \brief dispatch data to different pathes according to the indices and loads
 *
 * \param in_data shape: [cell_num, cell_size]
 * \param route_indices shape: [cell_num, path_num]
 * \param loads shape: [path_num]
 * \param gates (optional) shape: [cell_num, path_num]
 * \param routing_dim available dimensions: [1, 2]
 * \param auto_pad if true, pad each path to load limit (if load_limit is not set, use max of loads)
 * \param load_limit load limit
 */
std::vector<::torch::Tensor> dispatch_with_indices_and_loads(
    const ::torch::Tensor& in_data /*[cell_num x cell_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/,
    const bool& max_path_padding = false,
    const ::c10::optional<int>& cell_num_per_path = {} /* load limit*/,
    const bool& is_1d_routing = true /* available dimensions: [1, 2] */,
    const bool& tag_generating = false,
    const ::c10::optional<::torch::Tensor>& tags = {},
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
  ::torch::Tensor old_cell_tags;
  ::torch::Tensor new_cell_tags;

  if (tag_generating) {
    AT_ASSERT(tags.has_value());
    old_cell_tags = tags.value();
    new_cell_tags = ::torch::zeros({total_load}, route_indices.options());
  }

  auto out_shape = in_data.sizes().vec();
  out_shape[0] = total_load;
  auto out_data = ::torch::zeros(out_shape, in_data.options());

  if (data_type == ::torch::kFloat32) {
    router::DispatchWithIndicesAndLoads<float>(
        in_data.data_ptr(), out_data.data_ptr(), gates_data_ptr, route_indices.data_ptr<int>(),
        cuda_loads.data_ptr<int>(), cell_num, cell_size, path_num, cell_num_per_path_value,
        is_1d_routing, is_dst_index, at::cuda::getDefaultCUDAStream().stream());
  } else if (data_type == ::torch::kFloat16) {
    router::DispatchWithIndicesAndLoads<__half2>(
        in_data.data_ptr(), out_data.data_ptr(), gates_data_ptr, route_indices.data_ptr<int>(),
        cuda_loads.data_ptr<int>(), cell_num, cell_size, path_num, cell_num_per_path_value,
        is_1d_routing, is_dst_index, at::cuda::getDefaultCUDAStream().stream());
  } else {
    AT_ERROR("Unsupported data type: ", data_type);
  }
  if (tag_generating) {
    return {out_data};
  }

  return {out_data};
}

::torch::Tensor dispatch_with_dst_indices_1d(
    const ::torch::Tensor& in_data /*[cell_num x cell_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const bool& auto_pad = false,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);

  int cell_num = in_data.size(0);
  int cell_size = in_data.numel() / cell_num;
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
                                          capacity, cell_num, cell_size, path_num,
                                          at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor padded_dispatch_with_dst_indices_1d(
    const ::torch::Tensor& in_data /*[cell_num x cell_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const int& pad_size,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);

  int cell_num = in_data.size(0);
  int cell_size = in_data.numel() / cell_num;
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
                                          capacity, cell_num, cell_size, path_num,
                                          at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor dispatch_with_dst_indices_2d(
    const ::torch::Tensor& in_data /*[cell_num x cell_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const bool& auto_pad = false) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  CHECK_ON_CUDA(loads);

  int cell_num = in_data.size(0);
  int path_num = route_indices.size(1);
  int cell_size = in_data.numel() / (cell_num * path_num);

  int total_load = 0;
  int capacity = 0;
  if (auto_pad) {
    capacity = loads.max().item<int>();
    total_load = capacity * path_num;
  } else {
    total_load = loads.sum().item<int>();
  }

  auto in_data_to_be_route = in_data.transpose(0, 1).contiguous();

  auto out_shape = in_data.sizes().vec();
  out_shape[1] = total_load;
  at::IntArrayRef out_shape_ref(out_shape.data() + 1, out_shape.data() + out_shape.size());
  auto out_data = ::torch::zeros(out_shape_ref, in_data.options());

  router::DispatchWithDstIndices2D<float>(in_data_to_be_route.data_ptr(), out_data.data_ptr(),
                                          route_indices.data_ptr<int>(), loads.data_ptr<int>(),
                                          capacity, cell_num, cell_size, path_num,
                                          at::cuda::getDefaultCUDAStream().stream());
  return out_data;
}

::torch::Tensor combine_with_src_indices(
    const ::torch::Tensor& in_data /*[load*path_num x cell_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const bool& auto_pad = false,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/,
    const ::c10::optional<::torch::Tensor>& out_data = {} /*[cell_num x cell_size]*/) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  ::torch::Tensor cuda_loads;
  if (!loads.is_cuda()) {
    cuda_loads = loads.cuda();
  } else {
    cuda_loads = loads;
  }

  int cell_num = route_indices.size(0);
  int cell_size = in_data.numel() / in_data.size(0);
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
        cuda_loads.data_ptr<int>(), capacity, cell_num, cell_size, path_num,
        at::cuda::getDefaultCUDAStream().stream());
  } else {
    out_shape[0] = cell_num;
    out_data_t = ::torch::zeros(out_shape, in_data.options());
    CHECK_ON_CUDA(out_data_t);
    router::CombineWithSrcIndices<float>(in_data.data_ptr(), out_data_t.data_ptr(), gates_data_ptr,
                                         route_indices.data_ptr<int>(), cuda_loads.data_ptr<int>(),
                                         capacity, cell_num, cell_size, path_num,
                                         at::cuda::getDefaultCUDAStream().stream());
  }
  return out_data_t;
}

}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_indices_and_loads", &brt::backend::torch::generate_indices_and_loads,
        "Generate indices and loads for all paths, loads can be padded or throttled by supported "
        "capacities",
        pybind11::arg("hot_mask"), pybind11::arg("supported_capacities") = pybind11::none(),
        pybind11 ::arg("capacity_padding") = false, pybind11::arg("is_dst_index") = true,
        pybind11::arg("load_on_cpu") = false);
  m.def("convert_index_format", &brt::backend::torch::convert_index_format,
        "convert indices to the new index format", pybind11::arg("origin_indices"),
        pybind11::arg("loads"), pybind11::arg("dst_to_src"));
  m.def("dispatch_with_dst_indices_1d", &brt::backend::torch::dispatch_with_dst_indices_1d,
        "Route data with local indices", pybind11::arg("in_data"), pybind11::arg("route_indices"),
        pybind11::arg("loads"), pybind11::arg("auto_pad") = false,
        pybind11::arg("gates") = pybind11::none());
  m.def("dispatch_with_indices_and_loads", &brt::backend::torch::dispatch_with_indices_and_loads,
        "Route data with indices and loads, indices can be in dst or src format",
        pybind11::arg("in_data"), pybind11::arg("route_indices"), pybind11::arg("loads"),
        pybind11::arg("gates") = pybind11::none(), pybind11::arg("max_path_padding") = false,
        pybind11::arg("cell_num_per_path") = pybind11::none(),
        pybind11::arg("is_1d_routing") = true, pybind11::arg("tag_generating") = false,
        pybind11::arg("tags") = pybind11::none(), pybind11::arg("is_dst_index") = true);
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