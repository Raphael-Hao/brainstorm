/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/distributed/asymmetry.h>

#include "./nccl_manager.h"
#include "./torch.h"

namespace brt {
namespace backend {
namespace torch {

static ::torch::Tensor make_nccl_unique_id(const int& world_rank) {
  ::torch::Tensor nccl_unique_id_t = ::torch::empty(
      {sizeof(ncclUniqueId)}, ::torch::TensorOptions().dtype(::torch::kInt8).device(::torch::kCPU));
  CHECK_EQ(nccl_unique_id_t.nbytes(), sizeof(ncclUniqueId));
  if (world_rank == 0) {
    ncclUniqueId nccl_unique_id;
    CHECK_EQ(0, ncclGetUniqueId(&nccl_unique_id));
    memcpy((void*)nccl_unique_id_t.data_ptr(), &nccl_unique_id, sizeof(ncclUniqueId));
  }
  return nccl_unique_id_t;
}

static void init_nccl(::torch::Tensor unique_id, const int& world_rank,
                      const int& world_size, const int& event_num) {
  static bool initialized = false;
  auto unique_it_cpu = unique_id.to(::torch::kCPU);
  if (!initialized) {
    NcclManager& manager = NcclManager::GetManager();
    auto nccl_stream = at::cuda::getStreamFromPool();
    manager.Init(unique_it_cpu, world_rank, world_size, event_num);
    initialized = true;
  } else {
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
  // Construct the send size and recv size
  auto send_sizes_cpu = send_sizes.cpu();
  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());
  // Calculate the total size in byte
  int total_size_in_byte = in_data.numel() * in_data.element_size();
  // Calculate the size of each grainularity in byte
  int grain_size_in_byte = total_size_in_byte / in_data.size(0);
  // Calculate the slice and per slice size in byte
  int slice_num = send_sizes_cpu.numel();
  int slice_size_in_byte = total_size_in_byte / slice_num;

  auto& manager = NcclManager::GetManager();
  ::torch::Tensor out_data =
      ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);
  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(in_data);
  manager.RecordStorage(out_data);
  distributed::AsymmetryAllToAll(in_data.data_ptr(), out_data.data_ptr(), send_sizes_vec,
                                 recv_sizes_vec, grain_size_in_byte, slice_size_in_byte, slice_num,
                                 manager.GetComm(), manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  return out_data;
}
}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_nccl_unique_id", &brt::backend::torch::make_nccl_unique_id, "make nccl unique id",
        py::arg("world_rank"));
  m.def("init_nccl", &brt::backend::torch::init_nccl, "init nccl", py::arg("unique_id"),
        py::arg("world_rank"), py::arg("world_size"), py::arg("event_num") = 1);
  m.def("asymmetry_all_to_all", &brt::backend::torch::asymmetry_all_to_all, "asymmetry all to all",
        py::arg("in_data"), py::arg("send_sizes"), py::arg("recv_sizes"));
}