/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/distributed/asymmetry.h>
#include <brt/distributed/manager.h>

#include "./torch.h"

namespace brt {
namespace backend {
namespace torch {

static void init_nccl(::torch::Tensor unique_id_tensor, const int& world_rank,
                      const int& world_size) {
  static bool initialized = false;
  if (!initialized) {
    distributed::NCCLManager& manager = distributed::NCCLManager::get_manager();
    auto nccl_stream = at::cuda::getStreamFromPool();
    manager.init(unique_id_tensor.data_ptr(), world_rank, world_size, nccl_stream);
    initialized = true;
  }
}
static ::torch::Tensor asymmetry_all_to_all(const ::torch::Tensor& in_data,
                                            const ::torch::Tensor& send_sizes,
                                            const ::torch::Tensor& recv_sizes) {
  CHECK_ON_CPU(send_sizes);
  CHECK_ON_CPU(recv_sizes);
  CHECK_EQ(send_sizes.numel(), recv_sizes.numel());

  int total_size_in_byte = in_data.numel() * in_data.element_size();
  int grain_size_in_byte = total_size_in_byte / in_data.size(0);
  int slice_num = send_sizes.numel();
  int slice_size_in_byte = total_size_in_byte / slice_num;

  ::torch::Tensor out_data =
      ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);
  std::vector<int> send_sizes_vec(send_sizes.data_ptr<int>(),
                                  send_sizes.data_ptr<int>() + send_sizes.numel());
  std::vector<int> recv_sizes_vec(recv_sizes.data_ptr<int>(),
                                  recv_sizes.data_ptr<int>() + recv_sizes.numel());
  distributed::AsymmetryAllToAll(in_data.data_ptr(), out_data.data_ptr(), send_sizes_vec,
                                 recv_sizes_vec, grain_size_in_byte, slice_size_in_byte, slice_num);

  return out_data;
}
}  // namespace torch
}  // namespace backend
}  // namespace brt
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_nccl", &brt::backend::torch::init_nccl, "init nccl");
  m.def("asymmetry_all_to_all", &brt::backend::torch::asymmetry_all_to_all, "asymmetry all to all");
}