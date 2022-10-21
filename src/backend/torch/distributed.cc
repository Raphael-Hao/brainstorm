/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/distributed/collective.h>
#include <brt/distributed/local_reorder.h>

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

static void init_nccl(::torch::Tensor unique_id, const int& world_rank, const int& world_size) {
  static bool initialized = false;
  auto unique_it_cpu = unique_id.to(::torch::kCPU);
  if (!initialized) {
    NcclManager& manager = NcclManager::GetManager();
    auto nccl_stream = at::cuda::getStreamFromPool();
    manager.Init(unique_it_cpu, world_rank, world_size);
    initialized = true;
  } else {
    LOG(WARNING) << "NCCL has been initialized, Reinitilizing it.";
  }
}

static std::pair<::torch::Tensor, ::torch::Tensor> locality_reorder(const ::torch::Tensor& loads,
                                                                    const int& world_size) {
  CHECK_ON_CUDA(loads);
  ::torch::Tensor reordered_loads = ::torch::empty_like(loads, loads.options());
  ::torch::Tensor reorder_indices = ::torch::empty(world_size, loads.options());
  brt::distributed::LocalityReorder(
      loads.data_ptr<int>(), world_size, reorder_indices.data_ptr<int>(),
      reordered_loads.data_ptr<int>(), at::cuda::getCurrentCUDAStream().stream());
  return {reorder_indices, reordered_loads};
}

static std::vector<::torch::Tensor> asymmetry_all_to_all(const ::torch::Tensor& in_data,
                                                         const ::torch::Tensor& send_sizes,
                                                         bool locality = false) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();

  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(send_sizes);

  ::torch::Tensor recv_sizes = ::torch::empty_like(send_sizes, send_sizes.options());

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(send_sizes);
  manager.RecordStorage(recv_sizes);

  distributed::AllToAll(send_sizes.data_ptr(), recv_sizes.data_ptr(),
                        send_sizes.nbytes() / send_sizes.size(0), send_sizes.numel(),
                        manager.GetComm(), manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  auto send_sizes_cpu = send_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor all_recv_sizes;
  if (locality) {
    if (world_rank == 0) {
      all_recv_sizes =
          ::torch::empty({send_sizes.numel() * manager.GetWorldSize()}, send_sizes.options());
      manager.RecordStorage(all_recv_sizes);
    }
    manager.StartContext();
    manager.WaitEvent(1);
    distributed::Gather(recv_sizes.data_ptr(), all_recv_sizes.data_ptr(), recv_sizes.nbytes(), 0,
                        world_rank, world_size, manager.GetComm(), manager.GetStream());
    manager.RecordEvent(1);
    manager.EndContext();
  }

  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());

  // Calculate the total size in byte
  int total_size_in_byte = in_data.numel() * in_data.element_size();
  // Calculate the size of each grainularity in byte
  int grain_size_in_byte = total_size_in_byte / in_data.size(0);
  // Calculate the slice and per slice size in byte
  int slice_num = send_sizes_cpu.numel();
  int slice_size_in_byte = total_size_in_byte / slice_num;

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

  if (locality) {
    manager.ExternalWaitEvent(1, at::cuda::getCurrentCUDAStream());
    auto all_recv_sizes_cpu = all_recv_sizes.cpu();
    std::vector<int> all_recv_sizes_vec(
        all_recv_sizes_cpu.data_ptr<int>(),
        all_recv_sizes_cpu.data_ptr<int>() + all_recv_sizes_cpu.numel());
    ::torch::Tensor new_all_recv_sizes;
    std::vector<int> reorder_indices;
    for (int i = 0; i < world_size; i++) {
      auto max_element = std::max_element(all_recv_sizes_vec.begin(), all_recv_sizes_vec.end());
      int current_rank_index = (max_element - all_recv_sizes_vec.begin()) / world_size;
      int original_rank_index = (max_element - all_recv_sizes_vec.begin()) % world_size;
      std::fill_n(all_recv_sizes_vec.begin() + current_rank_index * world_size, world_size, -1);
      int new_rank_index = i;
      reorder_indices.emplace_back(original_rank_index);
    }
  }

  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());
  return {out_data, recv_sizes};
}
}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_nccl_unique_id", &brt::backend::torch::make_nccl_unique_id, "make nccl unique id",
        pybind11::arg("world_rank"));
  m.def("init_nccl", &brt::backend::torch::init_nccl, "init nccl", pybind11::arg("unique_id"),
        pybind11::arg("world_rank"), pybind11::arg("world_size"));
  m.def("locality_reorder", &brt::backend::torch::locality_reorder, "locality reorder",
        pybind11::arg("loads"), pybind11::arg("wolrd_size"));
  m.def("asymmetry_all_to_all", &brt::backend::torch::asymmetry_all_to_all, "asymmetry all to all",
        pybind11::arg("in_data"), pybind11::arg("send_sizes"), pybind11::arg("locality") = false);
}