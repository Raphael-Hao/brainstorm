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

std::pair<::torch::Tensor, ::torch::Tensor> throttle_hotmask(
    const ::torch::Tensor& hotmask /*[cell_num, path_num]*/,
    const ::torch::Tensor& prefix /*[path_num]*/,
    const ::torch::Tensor& threshold /*[path_num]*/) {
  CHECK_ON_CUDA(hotmask);
  CHECK_ON_CUDA(prefix);
  CHECK_ON_CUDA(threshold);
  CHECK_EQ(hotmask.size(1), prefix.size(0));
  CHECK_EQ(hotmask.size(1), threshold.size(0));
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(hotmask));

  ::torch::Tensor throttled_mask = ::torch::zeros_like(hotmask, hotmask.options());
  router::ThrottleHotmask(hotmask.data_ptr<int>(), throttled_mask.data_ptr<int>(),
                          prefix.data_ptr<int>(), threshold.data_ptr<int>(), hotmask.size(0),
                          hotmask.size(1), at::cuda::getDefaultCUDAStream().stream());
  return {throttled_mask, prefix};
}

/*!
 * \brief generate indices and loads according to hot_mask
 *
 * \param hot_mask shape: [cell_num, path_num]
 * \param supported_capacities (optional) shape: [supported_capacity_num]
 *                            if not empty, we will check if each path's load is below any
 *                            of the supported capacities. if not, we will drop the cells
 *                            out of the supported capacities.
 * \param capacity_padding (optional, default false) if true, we will pad the cells of the
 *                         path with load below the mapped capacity to the capacity.
 * \param is_tag_index (optional, default false) if true, generate indices which contains
 *                     tag of each cell for each path, otherwise generate indices which
 *                     contains new seat of each cell in each path.
 *
 */
std::pair<::torch::Tensor, ::torch::Tensor> generate_indices_and_loads(
    const ::torch::Tensor& hot_mask /*[cell_num, path_num]*/,
    const ::c10::optional<::torch::Tensor>& supported_capacities = {},
    const bool& capacity_padding = false,
    const bool& path_wise_padding = false,
    const bool& is_tag_index = false) {
  CHECK_ON_CUDA(hot_mask);
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(hot_mask));

  auto cell_num = hot_mask.size(0);
  auto path_num = hot_mask.size(1);
  if (hot_mask.numel() == 0) {
    return {::torch::zeros({0, path_num}, hot_mask.options()),
            ::torch::zeros({path_num}, hot_mask.options())};
  }

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
      supported_capacities_data_ptr, supported_capacity_num, capacity_padding, path_wise_padding,
      is_tag_index, at::cuda::getDefaultCUDAStream().stream());
  return {indices, loads};
}

/*!
 * \brief convert index format from tag to seat or from seat to tag
 *
 * \param origin_indices shape: [cell_num, path_num]
 * \param loads shape: [path_num]
 * \param is_to_tag if true, convert from seat to tag, otherwise convert from tag to seat
 */

::torch::Tensor convert_index_format(
    const ::torch::Tensor& origin_indices /*[cell_num x path_num]*/,
    const ::torch::Tensor& loads /*[path_num]*/,
    const bool& is_to_tag) {
  CHECK_ON_CUDA(origin_indices);
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(origin_indices));

  ::torch::Tensor cuda_loads;
  if (!loads.is_cuda() && !is_to_tag) {
    cuda_loads = loads.to(origin_indices.device(), true);
  } else {
    cuda_loads = loads;
  }
  ::torch::Tensor new_indices = ::torch::zeros_like(origin_indices, origin_indices.options());
  auto cell_num = origin_indices.size(0);
  auto path_num = origin_indices.size(1);
  router::ConvertIndexFormat(origin_indices.data_ptr<int>(), new_indices.data_ptr<int>(),
                             cuda_loads.data_ptr<int>(), cell_num, path_num, is_to_tag,
                             at::cuda::getDefaultCUDAStream().stream());
  return new_indices;
}

/*!
 * \brief dispatch data to different pathes according to the indices and loads
 *
 * \param in_data shape: [cell_num, *cell_shape]
 * \param route_indices shape: [cell_num, path_num]
 * \param loads shape: [path_num]
 * \param gates (optional) shape: [cell_num, path_num]
 * \param tag_generating (optional, default false) if true, we will generate tags for each path's
 *                       cells.
 * \param tags (optional) shape: [cell_num, path_num] original tags of each cell for in_data
 * \param max_path_padding (optional, default false) if true, we will pad each path equally to the
 *                         max path length.
 * \param max_path_load (optional) will be used as the max load of all paths if max_path_padding is
 *                      true.
 * \param is_1d_routing (optional, default true) if true, we will use 1d routing, otherwise use 2d
 *                     routing.
 * \param is_tag_index (optional, default false) if true, we will use tag index, otherwise use seat
 *                    index.
 */
std::vector<::torch::Tensor> dispatch_with_indices_and_loads(
    const ::torch::Tensor& in_data,
    const ::torch::Tensor& route_indices,
    const ::torch::Tensor& loads,
    const ::c10::optional<::torch::Tensor>& gates = {},
    const bool& tag_generating = false,
    const ::c10::optional<::torch::Tensor>& tags = {},
    const bool& max_path_padding = false,
    const int& max_path_load = 0,
    const bool& is_1d_routing = true,
    const bool& is_tag_index = false) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(in_data));
  auto data_type = in_data.dtype();

  int cell_num = in_data.size(0);
  int path_num = route_indices.size(1);

  if (cell_num == 0) {
    auto out_shape = in_data.sizes().vec();
    at::IntArrayRef out_shape_ref;
    if (is_1d_routing) {
      out_shape[0] = 0;
      out_shape_ref = at::IntArrayRef(out_shape.data(), out_shape.data() + out_shape.size());
    } else {
      if (out_shape.size() == 2) {
        out_shape.push_back(1);
      }
      out_shape[1] = 0;
      out_shape_ref = at::IntArrayRef(out_shape.data() + 1, out_shape.data() + out_shape.size());
    }
    if (tag_generating) {
      return {::torch::zeros(out_shape_ref, in_data.options()),
              ::torch::zeros({0}, route_indices.options())};
    }
    return {::torch::zeros(out_shape_ref, in_data.options())};
  }

  int cell_size = in_data.numel() / cell_num;

  if (!is_1d_routing) {
    cell_size = cell_size / path_num;
  }

  if (data_type == ::torch::kFloat16) {
    TORCH_INTERNAL_ASSERT(cell_size % 2 == 0, "cell_size must be even when data type is float16");
    cell_size = cell_size / 2;
  }

  ::torch::Tensor cuda_loads;

  if (!loads.is_cuda()) {
    cuda_loads = loads.to(in_data.device(), true);
  } else {
    cuda_loads = loads;
  }

  // calculate total_load.
  int total_load = 0;
  int max_path_load_value = 0;
  if (max_path_padding) {
    if (max_path_load > 0) {
      max_path_load_value = max_path_load;
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
    if (out_shape.size() == 2) {
      out_shape.push_back(1);
    }
    out_shape[1] = total_load;
    out_shape_ref = at::IntArrayRef(out_shape.data() + 1, out_shape.data() + out_shape.size());
  }
  auto out_data = ::torch::zeros(out_shape_ref, in_data.options());

  // process gates, if dtype is half, repeat each element to 2 times. In CUDA kernel, we are using
  // half2 operator to calculate output.
  void* gates_data_ptr = nullptr;
  if (gates.has_value()) {
    CHECK(is_1d_routing);
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
      old_tags = ::torch::arange(1, cell_num + 1, route_indices.options());
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
        cell_num, cell_size, path_num, max_path_load_value, is_1d_routing, is_tag_index,
        at::cuda::getDefaultCUDAStream().stream());
  } else if (data_type == ::torch::kFloat16) {
    router::DispatchWithIndicesAndLoads<__half2>(
        in_data_to_be_route.data_ptr(), out_data.data_ptr(), gates_data_ptr,
        route_indices.data_ptr<int>(), cuda_loads.data_ptr<int>(), old_tags_ptr, new_tags_ptr,
        cell_num, cell_size, path_num, max_path_load_value, is_1d_routing, is_tag_index,
        at::cuda::getDefaultCUDAStream().stream());
  } else {
    TORCH_CHECK_NOT_IMPLEMENTED(data_type == ::torch::kFloat32 || data_type == ::torch::kFloat16,
                                "data type is not supported");
  }
  if (tag_generating) {
    return {out_data, new_tags};
  }
  return {out_data};
}

/*!
 * \brief Split fused cells to paths.
 *
 * \param in_data The input data for all paths.
 * \param loads The loads of all paths.
 * \param max_path_padding Whether the input data is padded to max path load.
 * \param is_load_split Whether the load should be split to each path.
 * \param is_tag_split Whether the tag should be split to each path.
 * \param tags The tags of all paths.
 */
std::vector<std::vector<::torch::Tensor>> split_fused_cells_to_paths(
    const ::torch::Tensor& in_data,
    const ::torch::Tensor& loads,
    const bool& max_path_padding = false,
    bool is_load_split = false,
    const bool& is_tag_split = false,
    const c10::optional<::torch::Tensor>& tags = {}) {
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(in_data));
  int path_num = loads.numel();
  if (is_tag_split) {
    is_load_split = true;
    CHECK(tags.has_value());
  }
  std::vector<::torch::Tensor> in_data_list;
  std::vector<::torch::Tensor> loads_list;
  std::vector<::torch::Tensor> tags_list;

  if (max_path_padding) {
    int max_path_load = in_data.size(0) / path_num;
    for (int i = 0; i < path_num; ++i) {
      in_data_list.emplace_back(in_data.narrow(0, i * max_path_load, max_path_load));
      if (is_load_split) {
        loads_list.emplace_back(loads.narrow(0, i, 1));
      }
      if (is_tag_split) {
        tags_list.emplace_back(tags.value().narrow(0, i * max_path_load, max_path_load));
      }
    }
  } else {
    ::torch::Tensor cpu_loads = loads.cpu();
    int load_start = 0;
    for (int i = 0; i < path_num; ++i) {
      int load = cpu_loads[i].item<int>();
      in_data_list.emplace_back(in_data.narrow(0, load_start, load));
      if (is_load_split) {
        loads_list.emplace_back(loads.narrow(0, i, 1));
      }
      if (is_tag_split) {
        tags_list.emplace_back(tags.value().narrow(0, load_start, load));
      }
      load_start += load;
    }
  }
  if (is_tag_split) {
    return {in_data_list, loads_list, tags_list};
  } else if (is_load_split) {
    return {in_data_list, loads_list};
  } else {
    return {in_data_list};
  }
}

/*!
 * \brief fuse split cells from paths into a single tensor.
 *
 * \param in_data The input data for all paths.
 * \param is_load_fuse (optional, default false) Whether the load should be fused to a single
 *                     tensor.
 * \param is_tag_fuse (optional, default false) Whether the tag should be fused to a single
 *                   tensor.
 * \param loads (optional) The loads of all paths.
 * \param tags (optional) The tags of all paths.
 */
std::vector<::torch::Tensor> fuse_split_cells_from_paths(
    const std::vector<::torch::Tensor>& in_data,
    bool is_load_fuse = false,
    const bool& is_tag_fuse = false,
    const c10::optional<std::vector<::torch::Tensor>>& loads = {},
    const c10::optional<std::vector<::torch::Tensor>>& tags = {}) {
  auto out_data = ::torch::cat(in_data, 0);
  if (is_load_fuse) {
    CHECK(loads.has_value());
    auto out_loads = ::torch::cat(loads.value(), 0);
    return {out_data, out_loads};
  }
  if (is_tag_fuse) {
    CHECK(tags.has_value());
    auto out_tags = ::torch::cat(tags.value(), 0);
    auto result = ::torch::_unique(out_tags, true, true);
    auto new_tags = std::get<0>(result);
    auto global_seat_indices = std::get<1>(result);
    return {out_data, new_tags, global_seat_indices};
  }
  return {out_data};
}

// TODO add support for predefined output size
std::vector<::torch::Tensor> combine_with_indices_and_loads(
    const ::torch::Tensor& in_data /*[load*path_num x cell_size]*/,
    const ::torch::Tensor& route_indices /*[cell_num x path_num]*/,
    const ::c10::optional<::torch::Tensor>& loads = {} /*[path_num]*/,
    const ::c10::optional<::torch::Tensor>& gates = {} /*[cell_num x path_num]*/,
    const ::c10::optional<::torch::Tensor>& out_data = {} /*[cell_num x cell_size]*/,
    const bool& max_path_padding = false,
    const bool& ever_padded = true,
    const bool& is_tag_index = false,
    const ::c10::optional<::torch::Tensor>& tags = {}) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(route_indices);
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(in_data));

  int cell_num = route_indices.size(0);
  if (cell_num == 0) {
    if (!is_tag_index) {
      if (out_data.has_value()) {
        return {out_data.value()};
      } else {
        return {::torch::empty_like(in_data)};
      }
    } else {
      return {::torch::empty_like(in_data), ::torch::empty_like(tags.value()),
              ::torch::zeros_like(tags.value())};
    }
  }
  int cell_size = in_data.numel() / in_data.size(0);
  // TODO need to check if cell_num==1 is equivalent to only one cell to be combined when we use tag
  // index
  if (cell_num == 1) {
    if (!is_tag_index) {
      if (out_data.has_value()) {
        return {out_data.value() + in_data};
      } else {
        return {in_data};
      }
    } else {
      return {in_data, tags.value(), ::torch::ones_like(tags.value())};
    }
  }
  ::torch::Tensor out_data_t;
  if (!is_tag_index) {
    CHECK(loads.has_value());
    ::torch::Tensor cuda_loads = loads.value().to(in_data.device(), true);

    int path_num = route_indices.size(1);
    void* gates_data_ptr = nullptr;
    if (gates.has_value()) {
      CHECK_ON_CUDA(gates.value());
      gates_data_ptr = gates.value().data_ptr();
    }

    auto out_shape = in_data.sizes().vec();

    int max_path_load = 0;
    if (max_path_padding) {
      max_path_load = out_shape[0] / path_num;
    }

    if (out_data.has_value()) {
      CHECK_ON_CUDA(out_data.value());
      out_data_t = out_data.value();
      router::CombineWithIndicesAndLoads<float>(
          in_data.data_ptr(), out_data_t.data_ptr(), gates_data_ptr, route_indices.data_ptr<int>(),
          cuda_loads.data_ptr<int>(), nullptr, nullptr, cell_num, cell_size, path_num,
          max_path_load, true, false, at::cuda::getDefaultCUDAStream().stream());
    } else {
      out_shape[0] = cell_num;
      out_data_t = ::torch::zeros(out_shape, in_data.options());
      CHECK_ON_CUDA(out_data_t);
      router::CombineWithIndicesAndLoads<float>(
          in_data.data_ptr(), out_data_t.data_ptr(), gates_data_ptr, route_indices.data_ptr<int>(),
          cuda_loads.data_ptr<int>(), nullptr, nullptr, cell_num, cell_size, path_num,
          max_path_load, false, false, at::cuda::getDefaultCUDAStream().stream());
    }
    return {out_data_t};
  } else {
    CHECK(tags.has_value());
    auto out_shape = in_data.sizes().vec();
    auto new_indices = route_indices.view({-1, 1}).repeat({1, cell_size}).view_as(in_data);
    auto tags_value = tags.value();
    auto tmp_cell_num = tags_value.size(0);
    if (out_data.has_value()) {
      out_data_t = out_data.value();
      CHECK_ON_CUDA(out_data_t);
      CHECK_EQ(out_data_t.size(0), tmp_cell_num);
      ::torch::scatter_reduce(out_data_t, 0, new_indices, in_data, "sum", false);
    } else {
      out_shape[0] = tmp_cell_num;
      out_data_t = ::torch::zeros(out_shape, in_data.options());
      out_data_t = ::torch::scatter_reduce(out_data_t, 0, new_indices, in_data, "sum");
      if (ever_padded && tags_value[0].item<int>() == 0) {
        out_data_t = out_data_t.narrow(0, 1, tmp_cell_num - 1);
        tags_value = tags_value.narrow(0, 1, tmp_cell_num - 1);
      }
    }
    ::torch::Tensor out_loads =
        ::torch::tensor({tags_value.numel()}, in_data.options().dtype(::torch::kInt32));
    return {out_data_t, tags_value, out_loads};
  }
}

}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("throttle_hotmask", &brt::backend::torch::throttle_hotmask,
        "Throttle hotmask by supported capacities", pybind11::arg("hot_mask"),
        pybind11::arg("prefix"), pybind11::arg("threshold"));
  m.def("generate_indices_and_loads", &brt::backend::torch::generate_indices_and_loads,
        "Generate indices and loads for all paths, loads can be padded or throttled by supported "
        "capacities",
        pybind11::arg("hot_mask"), pybind11::arg("supported_capacities") = pybind11::none(),
        pybind11 ::arg("capacity_padding") = false, pybind11::arg("path_wise_padding") = false,
        pybind11::arg("is_tag_index") = false);
  m.def("convert_index_format", &brt::backend::torch::convert_index_format,
        "convert indices to the new index format", pybind11::arg("origin_indices"),
        pybind11::arg("loads"), pybind11::arg("is_to_tag"));
  m.def("dispatch_with_indices_and_loads", &brt::backend::torch::dispatch_with_indices_and_loads,
        "Route data with indices and loads, indices can be in dst or src format",
        pybind11::arg("in_data"), pybind11::arg("route_indices"), pybind11::arg("loads"),
        pybind11::arg("gates") = pybind11::none(), pybind11::arg("tag_generating") = false,
        pybind11::arg("tags") = pybind11::none(), pybind11::arg("max_path_padding") = false,
        pybind11::arg("max_path_load") = 0, pybind11::arg("is_1d_routing") = true,
        pybind11::arg("is_tag_index") = false);
  m.def("split_fused_cells_to_paths", &brt::backend::torch::split_fused_cells_to_paths,
        pybind11::arg("in_data"), pybind11::arg("loads"), pybind11::arg("max_path_padding") = false,
        pybind11::arg("is_load_split") = false, pybind11::arg("is_tag_split") = false,
        pybind11::arg("tags") = pybind11::none());
  m.def("fuse_split_cells_from_paths", &brt::backend::torch::fuse_split_cells_from_paths,
        pybind11::arg("in_data"), pybind11::arg("is_load_fuse") = false,
        pybind11::arg("is_tag_fuse") = false, pybind11::arg("loads") = pybind11::none(),
        pybind11::arg("tags") = pybind11::none());
  m.def("combine_with_indices_and_loads", &brt::backend::torch::combine_with_indices_and_loads,
        "Route data back with indices and loads, indices can be in dst or src format",
        pybind11::arg("in_data"), pybind11::arg("route_indices"), pybind11::arg("loads"),
        pybind11::arg("gates") = pybind11::none(), pybind11::arg("out_data") = pybind11::none(),
        pybind11::arg("max_path_padding") = false, pybind11::arg("ever_padded") = true,
        pybind11::arg("is_tag_index") = false, pybind11::arg("tags") = pybind11::none());
}