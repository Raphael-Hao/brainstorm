/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <torch/script.h>
//

namespace brt {
namespace torchscript {
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor> ScatterRoute(
    torch::Tensor input, long router_kind, long route_num) {
  auto input_size = input.sizes();
  std::vector<torch::Tensor> route_results(route_num, torch::ones(input_size, torch::kFloat32));
  std::vector<torch::Tensor> route_indices(route_num, torch::ones(input_size, torch::kFloat32));
  torch::Tensor reverse_shape = torch::ones(input_size.size(), torch::kInt64);
  return std::make_tuple(route_results, route_indices, reverse_shape);
}

torch::Tensor GatherRoute(std::vector<torch::Tensor> inputs,
                          std::vector<torch::Tensor> reverse_indices, torch::Tensor reverse_shape,
                          long router_kind, long route_num) {
  auto output_dim = reverse_shape.numel();
  auto output_size_vec = std::vector<long>(output_dim, 0);
  for (auto i = 0; i < output_dim; i++) {
    output_size_vec[i] = reverse_shape[i].item().toInt();
  }
  torch::Tensor route_results = torch::ones(output_size_vec, torch::kFloat32);
  return route_results;
}
}  // namespace torchscript
}  // namespace brt

TORCH_LIBRARY(brt, m) {
  m.def("symbolic_scatter_route", brt::torchscript::ScatterRoute)
      .def("symbolic_gather_route", brt::torchscript::GatherRoute);
}