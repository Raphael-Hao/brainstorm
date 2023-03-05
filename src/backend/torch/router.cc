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
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(hot_mask));

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
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(origin_indices));

  ::torch::Tensor cuda_loads;
  if (!loads.is_cuda() && !dst_to_src) {
    cuda_loads = loads.to(loads.options().device(::torch::kCUDA), true);
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
    const ::torch::Tensor& in_data,
    const ::torch::Tensor& route_indices,
    const ::torch::Tensor& loads,
    const ::c10::optional<::torch::Tensor>& gates = {},
    const bool& tag_generating = false,
    const ::c10::optional<::torch::Tensor>& tags = {},
    const bool& max_path_padding = false,
    const ::c10::optional<int>& max_path_load = {},
    const bool& is_1d_routing = true,
    const bool& is_dst_index = true) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(in_data));
  auto data_type = in_data.dtype();

  int cell_num = in_data.size(0);
  int path_num = route_indices.size(1);
  int cell_size = in_data.numel() / cell_num;

  if (!is_1d_routing) {
    cell_size = cell_size / path_num;
  }

  if (data_type == ::torch::kFloat16) {
    AT_ASSERTM(cell_size % 2 == 0, "cell_size must be even when data type is float16");
    cell_size = cell_size / 2;
  }

  ::torch::Tensor cuda_loads;

  if (!loads.is_cuda()) {
    cuda_loads = loads.to(loads.options().device(::torch::kCUDA), true);
  } else {
    cuda_loads = loads;
  }

  // calculate total_load.
  int total_load = 0;
  int max_path_load_value = 0;
  if (max_path_padding) {
    if (max_path_load.has_value()) {
      max_path_load_value = max_path_load.value();
      total_load = max_path_load_value * path_num;
    } else {
      max_path_load_value = loads.max().item<int>();
      total_load = max_path_load_value * path_num;
    }
  } else {
    total_load = loads.sum().item<int>();
  }

  // process in_data and allocate the device Tensor out_data according to is_1d_routing.
  auto out_shape = in_data.sizes().vec();
  at::IntArrayRef out_shape_ref;
  auto in_data_to_be_route = in_data;
  if (is_1d_routing) {
    out_shape[0] = total_load;
    out_shape_ref = at::IntArrayRef(out_shape.data(), out_shape.data() + out_shape.size());
  } else {
    in_data_to_be_route = in_data.transpose(0, 1).contiguous();
    out_shape[1] = total_load;
    out_shape_ref = at::IntArrayRef(out_shape.data() + 1, out_shape.data() + out_shape.size());
  }
  auto out_data = ::torch::zeros(out_shape_ref, in_data.options());

  // process gates, if dtype is half, repeat each element to 2 times. In CUDA kernel, we are using
  // half2 operator to calculate output.
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

  // prcoess tags
  ::torch::Tensor old_tags;
  ::torch::Tensor new_tags;
  int* old_tags_ptr = nullptr;
  int* new_tags_ptr = nullptr;
  if (tag_generating) {
    if (!tags.has_value()) {
      old_tags = ::torch::arange(1, total_load, route_indices.options());
    } else {
      CHECK_ON_CUDA(tags.value());
      CHECK_EQ(tags.value().dtype(), route_indices.dtype());
      old_tags = tags.value();
    }
    new_tags = ::torch::zeros({total_load}, route_indices.options());
    old_tags_ptr = old_tags.data_ptr<int>();
    new_tags_ptr = new_tags.data_ptr<int>();
  }

  if (data_type == ::torch::kFloat32) {
    router::DispatchWithIndicesAndLoads<float>(
        in_data_to_be_route.data_ptr(), out_data.data_ptr(), gates_data_ptr,
        route_indices.data_ptr<int>(), cuda_loads.data_ptr<int>(), old_tags_ptr, new_tags_ptr,
        cell_num, cell_size, path_num, max_path_load_value, is_1d_routing, is_dst_index,
        at::cuda::getDefaultCUDAStream().stream());
  } else if (data_type == ::torch::kFloat16) {
    router::DispatchWithIndicesAndLoads<__half2>(
        in_data_to_be_route.data_ptr(), out_data.data_ptr(), gates_data_ptr,
        route_indices.data_ptr<int>(), cuda_loads.data_ptr<int>(), old_tags_ptr, new_tags_ptr,
        cell_num, cell_size, path_num, max_path_load_value, is_1d_routing, is_dst_index,
        at::cuda::getDefaultCUDAStream().stream());
  } else {
    AT_ERROR("Unsupported data type: ", data_type);
  }
  if (tag_generating) {
    return {out_data, new_tags};
  }
  return {out_data};
}

std::vector<std::vector<::torch::Tensor>> split_fused_cells_to_paths(
    const ::torch::Tensor& in_data,
    const ::torch::Tensor& loads,
    const bool& max_path_padding = false,
    const c10::optional<::torch::Tensor>& tags = {}) {
  int path_num = loads.numel();

  ::torch::Tensor tags_value;
  if (tags.has_value()) {
    tags_value = tags.value();
  }
  std::vector<::torch::Tensor> in_data_list;
  std::vector<::torch::Tensor> loads_list;
  std::vector<::torch::Tensor> tags_list;

  if (max_path_padding) {
    int max_path_load = in_data.size(0) / path_num;
    for (int i = 0; i < path_num; ++i) {
      in_data_list.push_back(in_data.narrow(0, i * max_path_load, max_path_load));
      loads_list.push_back(loads.narrow(0, i, 1));
      if (tags.has_value()) {
        tags_list.push_back(tags_value.narrow(0, i * max_path_load, max_path_load));
      }
    }
  } else {
    ::torch::Tensor cpu_loads = loads.cpu();
    int load_start = 0;
    for (int i = 0; i < path_num; ++i) {
      int load = cpu_loads[i].item<int>();
      in_data_list.push_back(in_data.narrow(0, load_start, load));
      loads_list.push_back(loads.narrow(0, i, 1));
      if (tags.has_value()) {
        tags_list.push_back(tags_value.narrow(0, load_start, load));
      }
      load_start += load;
    }
  }
  if (tags.has_value()) {
    return {in_data_list, loads_list, tags_list};
  } else {
    return {in_data_list, loads_list};
  }
}

std::vector<::torch::Tensor> fuse_split_cells_from_paths(
    const std::vector<::torch::Tensor>& in_data,
    const std::vector<::torch::Tensor>& loads,
    const bool& max_path_padding = false,
    const c10::optional<int>& max_path_load = {},
    const c10::optional<std::vector<::torch::Tensor>>& tags = {}) {
  return {in_data[0], loads[0]};
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
    cuda_loads = loads.to(loads.options().device(::torch::kCUDA), true);
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
  m.def("dispatch_with_indices_and_loads", &brt::backend::torch::dispatch_with_indices_and_loads,
        "Route data with indices and loads, indices can be in dst or src format",
        pybind11::arg("in_data"), pybind11::arg("route_indices"), pybind11::arg("loads"),
        pybind11::arg("gates") = pybind11::none(), pybind11::arg("tag_generating") = false,
        pybind11::arg("tags") = pybind11::none(), pybind11::arg("max_path_padding") = false,
        pybind11::arg("max_path_load") = pybind11::none(), pybind11::arg("is_1d_routing") = true,
        pybind11::arg("is_dst_index") = true);
  m.def("split_fused_cells_to_paths", &brt::backend::torch::split_fused_cells_to_paths,
        pybind11::arg("in_data"), pybind11::arg("loads"), pybind11::arg("max_path_padding") = false,
        pybind11::arg("tags") = pybind11::none());
  m.def("fuse_split_cells_from_paths", &brt::backend::torch::fuse_split_cells_from_paths,
        pybind11::arg("in_data"), pybind11::arg("loads"), pybind11::arg("max_path_padding") = false,
        pybind11::arg("max_path_load") = pybind11::none(),
        pybind11::arg("tags") = pybind11::none());
  m.def("combine_with_src_indices", &brt::backend::torch::combine_with_src_indices,
        "Route data back with dst indices", pybind11::arg("in_data"),
        pybind11::arg("route_indices"), pybind11::arg("loads"), pybind11::arg("auto_pad") = false,
        pybind11::arg("gates") = pybind11::none(), pybind11::arg("out_data") = pybind11::none());
}