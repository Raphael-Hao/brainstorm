/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <torch/script.h>
//

namespace brt {
namespace torchscript {

std::tuple<::torch::Tensor, ::torch::Tensor> TagRoute(const ::torch::Tensor& inputs) {
  ::torch::Tensor route_tags = ::at::arange(inputs.size(0), inputs.options());
  return std::make_tuple(inputs, route_tags);
}

std::tuple<std::vector<::torch::Tensor>, std::vector<::torch::Tensor>, long> ScatterRoute(
    const ::torch::Tensor& inputs, const ::torch::Tensor& tags, const long& route_num) {
  std::vector<::torch::Tensor> route_results(route_num,
                                             ::torch::empty_like(inputs, ::torch::kFloat32));
  std::vector<::torch::Tensor> route_tags(route_num,
                                          ::torch::ones(inputs.size(0), ::torch::kInt64));
  long loads = inputs.size(0);
  return std::make_tuple(route_results, route_tags, loads);
}

std::tuple<::torch::Tensor, ::torch::Tensor> GatherRoute(const std::vector<::torch::Tensor>& inputs,
                                                         const std::vector<::torch::Tensor>& tags,
                                                         const long& loads, const long& route_num) {
  auto cat_inputs = ::at::cat(inputs, 0);
  auto cat_tags = ::at::cat(tags, 0);
  return std::make_tuple(cat_inputs, cat_tags);
}

std::tuple<::torch::Tensor, ::torch::Tensor, ::torch::Tensor, ::torch::Tensor>
FusedPaddingScatterRoute(const ::torch::Tensor& input, const long& route_num,
                         const std::vector<long>& supported_capacities) {
  ::torch::Tensor reverse_shape = ::at::_shape_as_tensor(input);
  ::torch::Tensor route_tags = ::torch::randint(0, route_num, input.size(0));
  ::torch::Tensor route_results = ::torch::gather(input, 0, route_tags);
  ::torch::Tensor route_capacities = ::torch::empty(route_num, torch::kInt64);
  return std::make_tuple(route_results, route_tags, route_capacities, reverse_shape);
}

}  // namespace torchscript
}  // namespace brt

TORCH_LIBRARY(brt, m) {
  m.def("symbolic_tag_route", brt::torchscript::TagRoute)
      .def("symbolic_scatter_route", brt::torchscript::ScatterRoute)
      .def("symbolic_gather_route", brt::torchscript::GatherRoute);
}