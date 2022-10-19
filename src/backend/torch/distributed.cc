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
    distributed::NcclManager& manager = distributed::NcclManager::GetManager();
    auto nccl_stream = at::cuda::getStreamFromPool();
    manager.Init(unique_id_tensor.data_ptr(), world_rank, world_size, nccl_stream);
    initialized = true;
  }else{
    LOG(WARNING) << "NCCL has been initialized, Reinitilizing it.";
  }
}
static ::torch::Tensor asymmetry_all_to_all(const ::torch::Tensor& in_data,
                                            const ::torch::Tensor& send_sizes,
                                            const ::torch::Tensor& recv_sizes) {
  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(send_sizes);
  CHECK_ON_CUDA(recv_sizes);
  CHECK_EQ(send_sizes.numel(), recv_sizes.numel());
  auto send_sizes_cpu = send_sizes.cpu();
  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());
  int total_size_in_byte = in_data.numel() * in_data.element_size();
  int grain_size_in_byte = total_size_in_byte / in_data.size(0);
  int slice_num = send_sizes_cpu.numel();
  int slice_size_in_byte = total_size_in_byte / slice_num;

  ::torch::Tensor out_data =
      ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);

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