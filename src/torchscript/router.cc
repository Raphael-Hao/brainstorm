/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <torch/script.h>
//

namespace brt {
namespace torchscript {
torch::Tensor ScatterRouter(torch::Tensor input, long router_kind) {
  torch::Tensor output = torch::empty({0}, torch::kFloat32);
  return output.clone();
}
}  // namespace torchscript
}  // namespace brt

TORCH_LIBRARY(BRT, m) { m.def("scatter_router", brt::torchscript::ScatterRouter); }