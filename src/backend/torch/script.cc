/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <torch/script.h>
//

namespace brt {
namespace torchscript {
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor> ScatterRoute(
    const torch::Tensor& input, const long& router_kind, const long& route_num) {
  std::vector<torch::Tensor> route_results(route_num, torch::empty_like(input, torch::kFloat32));
  std::vector<torch::Tensor> route_indices(route_num, torch::ones(input.size(0), torch::kInt64));
  torch::Tensor reverse_shape = at::_shape_as_tensor(input);
  return std::make_tuple(route_results, route_indices, reverse_shape);
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

torch::Tensor GatherRoute(std::vector<torch::Tensor> inputs,
                          std::vector<torch::Tensor> reverse_indices, torch::Tensor reverse_shape,
                          long router_kind, long route_num) {
  // auto output_dim = reverse_shape.numel();
  // auto output_size_vec = std::vector<long>(output_dim, 0);
  // for (auto i = 0; i < output_dim; i++) {
  //   output_size_vec[i] = reverse_shape[i].item().toInt();
  // }
  std::vector<long> tensor_shape(reverse_shape.data_ptr<long>(),
                                 reverse_shape.data_ptr<long>() + reverse_shape.numel());
  torch::Tensor route_results = torch::ones(tensor_shape, torch::kFloat32);
  return route_results;
}
}  // namespace torchscript
}  // namespace brt

TORCH_LIBRARY(brt, m) {
  m.def("symbolic_scatter_route", brt::torchscript::ScatterRoute)
      .def("symbolic_gather_route", brt::torchscript::GatherRoute);
}