/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <torch/script.h>
//

namespace brt {
namespace torchscript {
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, long> ScatterRoute(
    const torch::Tensor& inputs, const long& router_kind, const long& route_num) {
  std::vector<torch::Tensor> route_results(route_num, torch::empty_like(inputs, torch::kFloat32));
  std::vector<torch::Tensor> route_indices(route_num, torch::ones(inputs.size(0), torch::kInt64));
  long loads = inputs.size(0);
  return std::make_tuple(route_results, route_indices, loads);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> FusedPaddingScatterRoute(
    const torch::Tensor& input, const long& router_kind, const long& route_num,
    const std::vector<long>& supported_capacities) {
  torch::Tensor reverse_shape = at::_shape_as_tensor(input);
  torch::Tensor route_indices = torch::randint(0, route_num, input.size(0));
  torch::Tensor route_results = torch::gather(input, 0, route_indices);
  torch::Tensor route_capacities = torch::empty(route_num, torch::kInt64);
  return std::make_tuple(route_results, route_indices, route_capacities, reverse_shape);
}

torch::Tensor GatherRoute(const std::vector<torch::Tensor>& inputs,
                          const std::vector<torch::Tensor>& reverse_indices, const long& loads,
                          const long& router_kind, const long& route_num) {
  std::vector<long> tensor_shape;
  auto& sample_input = inputs[0];
  for (auto& dim : sample_input.sizes()) {
    tensor_shape.push_back(dim);
  }
  tensor_shape[0] = loads;
  torch::Tensor route_results = torch::ones(tensor_shape, torch::kFloat32);
  return route_results;
}
}  // namespace torchscript
}  // namespace brt

TORCH_LIBRARY(brt, m) {
  m.def("symbolic_scatter_route", brt::torchscript::ScatterRoute)
      .def("symbolic_gather_route", brt::torchscript::GatherRoute);
}