/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <torch/script.h>
//

namespace brt {
namespace torchscript {

std::vector<::torch::Tensor> ScatterRoute(const ::torch::Tensor& inputs, const long& dst_num) {
  std::vector<::torch::Tensor> results(dst_num, ::torch::empty_like(inputs, ::torch::kFloat32));
  return results;
}

::torch::Tensor GatherRoute(const std::vector<::torch::Tensor>& inputs, const long& dst_num) {
  assert(static_cast<int>(inputs.size()) == dst_num);
  auto results = ::at::cat(inputs, 0);
  return results;
}

std::tuple<::torch::Tensor, ::torch::Tensor, ::torch::Tensor, ::torch::Tensor>
FusedPaddingScatterRoute(const ::torch::Tensor& input, const long& dst_num,
                         const std::vector<long>& supported_capacities) {
  ::torch::Tensor reverse_shape = ::at::_shape_as_tensor(input);
  ::torch::Tensor route_tags = ::torch::randint(0, dst_num, input.size(0));
  ::torch::Tensor route_results = ::torch::gather(input, 0, route_tags);
  ::torch::Tensor route_capacities = ::torch::empty(dst_num, torch::kInt64);
  return std::make_tuple(route_results, route_tags, route_capacities, reverse_shape);
}

}  // namespace torchscript
}  // namespace brt

TORCH_LIBRARY(brt, m) {
  m.def("symbolic_scatter_route", brt::torchscript::ScatterRoute)
      .def("symbolic_gather_route", brt::torchscript::GatherRoute);
}