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

}  // namespace torchscript
}  // namespace brt

TORCH_LIBRARY(brt, m) {
  m.def("symbolic_scatter_route", brt::torchscript::ScatterRoute)
      .def("symbolic_gather_route", brt::torchscript::GatherRoute);
}