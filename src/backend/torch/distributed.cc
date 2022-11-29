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
  nccl_unique_id_t = nccl_unique_id_t.to(::torch::kCUDA);
  CHECK_ON_CUDA(nccl_unique_id_t);
  return nccl_unique_id_t;
}

static void init_nccl(::torch::Tensor unique_id, const int& world_rank, const int& world_size) {
  static bool initialized = false;
  auto unique_it_cpu = unique_id.to(::torch::kCPU);
  CHECK_ON_CPU(unique_it_cpu);
  if (!initialized) {
    NcclManager& manager = NcclManager::GetManager();
    manager.Init(unique_it_cpu, world_rank, world_size);
    initialized = true;
  } else {
    LOG(WARNING) << "NCCL has been initialized, Reinitilizing it.";
  }
}

static std::pair<::torch::Tensor, ::torch::Tensor> locality_reorder(const ::torch::Tensor& loads,
                                                                    const int& world_size) {
  CHECK_ON_CUDA(loads);
  ::torch::Tensor reordered_loads =
      ::torch::empty_like(loads, loads.options(), ::torch::MemoryFormat::Contiguous);
  ::torch::Tensor reorder_indices =
      ::torch::empty(world_size, loads.options(), ::torch::MemoryFormat::Contiguous);
  distributed::LocalityReorder(loads.data_ptr<int>(), world_size, reorder_indices.data_ptr<int>(),
                               reordered_loads.data_ptr<int>(),
                               at::cuda::getCurrentCUDAStream().stream());
  return {reordered_loads, reorder_indices};
}

static std::pair<::torch::Tensor, ::torch::Tensor> group_locality_reorder(
    const ::torch::Tensor& loads, const int& world_size, const int& group_size = 1) {
  CHECK_ON_CUDA(loads);
  ::torch::Tensor reordered_loads =
      ::torch::empty_like(loads, loads.options(), ::torch::MemoryFormat::Contiguous);
  ::torch::Tensor reorder_indices =
      ::torch::empty(world_size, loads.options(), ::torch::MemoryFormat::Contiguous);
  distributed::GroupLocalityReorder(
      loads.data_ptr<int>(), group_size, world_size, reorder_indices.data_ptr<int>(),
      reordered_loads.data_ptr<int>(), at::cuda::getCurrentCUDAStream().stream());
  return {reordered_loads, reorder_indices};
}

static ::torch::Tensor exchange(const ::torch::Tensor& in_data,
                                const ::torch::Tensor& reorder_indices) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor reorder_indices_cpu = reorder_indices.to(::torch::kCPU);
  auto reorder_indices_cpu_ptr = reorder_indices_cpu.data_ptr<int>();
  int& src_rank = reorder_indices_cpu_ptr[world_rank];
  int dst_rank;
  for (int i = 0; i < world_size; ++i) {
    if (reorder_indices_cpu_ptr[i] == world_rank) {
      dst_rank = i;
      break;
    }
  }
  CHECK_ON_CUDA(in_data);

  ::torch::Tensor out_data =
      ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);
  // at::cuda::CUDACachingAllocator::recordStream(out_data.storage().data_ptr(),
  //                                              at::cuda::getCurrentCUDAStream());

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(in_data);
  manager.RecordStorage(out_data);
  distributed::Exchange(in_data.data_ptr(), out_data.data_ptr(), in_data.nbytes(), dst_rank,
                        src_rank, manager.GetComm(), manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());
  return out_data;
}

static std::vector<::torch::Tensor> batched_exchange(const std::vector<::torch::Tensor>& in_datas,
                                                     const ::torch::Tensor& reorder_indices) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor reorder_indices_cpu = reorder_indices.to(::torch::kCPU);
  auto reorder_indices_cpu_ptr = reorder_indices_cpu.data_ptr<int>();
  int& src_rank = reorder_indices_cpu_ptr[world_rank];
  int dst_rank;
  for (int i = 0; i < world_size; ++i) {
    if (reorder_indices_cpu_ptr[i] == world_rank) {
      dst_rank = i;
      break;
    }
  }
  std::vector<::torch::Tensor> out_datas;
  for (auto& in_data : in_datas) {
    CHECK_ON_CUDA(in_data);
    ::torch::Tensor out_data =
        ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);
    // at::cuda::CUDACachingAllocator::recordStream(out_data.storage().data_ptr(),
    //                                              at::cuda::getCurrentCUDAStream());
    manager.StartContext();
    manager.WaitEvent(0);
    manager.RecordStorage(in_data);
    manager.RecordStorage(out_data);
    distributed::Exchange(in_data.data_ptr(), out_data.data_ptr(), in_data.nbytes(), dst_rank,
                          src_rank, manager.GetComm(), manager.GetStream());
    manager.RecordEvent(0);
    manager.EndContext();
    manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());
    out_datas.push_back(out_data);
  }
  return out_datas;
}

static ::torch::Tensor reverse_exchange(const ::torch::Tensor& in_data,
                                        const ::torch::Tensor& reorder_indices) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor reorder_indices_cpu = reorder_indices.to(::torch::kCPU);
  auto reorder_indices_cpu_ptr = reorder_indices_cpu.data_ptr<int>();
  int& dst_rank = reorder_indices_cpu_ptr[world_rank];
  int src_rank;
  for (int i = 0; i < world_size; ++i) {
    if (reorder_indices_cpu_ptr[i] == world_rank) {
      src_rank = i;
      break;
    }
  }
  CHECK_ON_CUDA(in_data);

  ::torch::Tensor out_data =
      ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(in_data);
  manager.RecordStorage(out_data);
  distributed::Exchange(in_data.data_ptr(), out_data.data_ptr(), in_data.nbytes(), dst_rank,
                        src_rank, manager.GetComm(), manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());
  return out_data;
}

static std::vector<::torch::Tensor> batched_reverse_exchange(
    const std::vector<::torch::Tensor>& in_datas, const ::torch::Tensor& reorder_indices) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  CHECK_ON_CUDA(reorder_indices);

  ::torch::Tensor reorder_indices_cpu = reorder_indices.to(::torch::kCPU);
  auto reorder_indices_cpu_ptr = reorder_indices_cpu.data_ptr<int>();
  int& dst_rank = reorder_indices_cpu_ptr[world_rank];
  int src_rank;
  for (int i = 0; i < world_size; ++i) {
    if (reorder_indices_cpu_ptr[i] == world_rank) {
      src_rank = i;
      break;
    }
  }
  std::vector<::torch::Tensor> out_datas;
  for (auto& in_data : in_datas) {
    CHECK_ON_CUDA(in_data);

    ::torch::Tensor out_data =
        ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);

    manager.StartContext();
    manager.WaitEvent(0);
    manager.RecordStorage(in_data);
    manager.RecordStorage(out_data);
    distributed::Exchange(in_data.data_ptr(), out_data.data_ptr(), in_data.nbytes(), dst_rank,
                          src_rank, manager.GetComm(), manager.GetStream());
    manager.RecordEvent(0);
    manager.EndContext();
    manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());
    out_datas.push_back(out_data);
  }
  return out_datas;
}

static std::vector<::torch::Tensor> asymmetry_all_to_all(const ::torch::Tensor& in_data,
                                                         const ::torch::Tensor& send_sizes,
                                                         bool locality_aware = false) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();

  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(send_sizes);

  ::torch::Tensor recv_sizes =
      ::torch::empty_like(send_sizes, send_sizes.options(), ::torch::MemoryFormat::Contiguous);

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(send_sizes);
  manager.RecordStorage(recv_sizes);

  distributed::AllToAll(send_sizes.data_ptr(), recv_sizes.data_ptr(),
                        send_sizes.nbytes() / world_size, world_size, manager.GetComm(),
                        manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();

  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor all_recv_sizes;
  if (locality_aware) {
    if (world_rank == 0) {
      all_recv_sizes = ::torch::empty({send_sizes.numel() * manager.GetWorldSize()},
                                      send_sizes.options(), ::torch::MemoryFormat::Contiguous);
      manager.RecordStorage(all_recv_sizes);
    }
    manager.StartContext();
    manager.WaitEvent(1);
    if (world_rank == 0) {
      distributed::Gather(recv_sizes.data_ptr(), all_recv_sizes.data_ptr(), recv_sizes.nbytes(), 0,
                          world_rank, world_size, manager.GetComm(), manager.GetStream());
    } else {
      distributed::Gather(recv_sizes.data_ptr(), nullptr, recv_sizes.nbytes(), 0, world_rank,
                          world_size, manager.GetComm(), manager.GetStream());
    }
    manager.RecordEvent(1);
    manager.EndContext();
  }

  auto send_sizes_cpu = send_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());

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

  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor reorder_indices;
  ::torch::Tensor all_reordered_loads;
  ::torch::Tensor reordered_loads =
      ::torch::empty_like(send_sizes, send_sizes.options(), ::torch::MemoryFormat::Contiguous);

  if (locality_aware) {
    manager.ExternalWaitEvent(1, at::cuda::getCurrentCUDAStream());
    if (world_rank == 0) {
      auto reorder_results = locality_reorder(all_recv_sizes, world_size);
      all_reordered_loads = reorder_results.first;
      reorder_indices = reorder_results.second;
      manager.RecordStorage(all_reordered_loads);
    } else {
      reorder_indices =
          ::torch::empty_like(send_sizes, send_sizes.options(), ::torch::MemoryFormat::Contiguous);
    }
    manager.ExternalRecordEvent(1, at::cuda::getCurrentCUDAStream());
    manager.StartContext();
    manager.WaitEvent(1);
    manager.RecordStorage(reorder_indices);
    manager.RecordStorage(reordered_loads);
    if (world_rank == 0) {
      distributed::Scatter(all_reordered_loads.data_ptr(), reordered_loads.data_ptr(),
                           reordered_loads.nbytes(), 0, world_rank, world_size, manager.GetComm(),
                           manager.GetStream());
    } else {
      distributed::Scatter(nullptr, reordered_loads.data_ptr(), reordered_loads.nbytes(), 0,
                           world_rank, world_size, manager.GetComm(), manager.GetStream());
    }
    distributed::BroadCast(reorder_indices.data_ptr(), reorder_indices.data_ptr(),
                           reorder_indices.nbytes(), 0, manager.GetComm(), manager.GetStream());
    manager.RecordEvent(1);
    manager.EndContext();
    manager.ExternalWaitEvent(1, at::cuda::getCurrentCUDAStream());
  }

  if (locality_aware) {
    return {out_data, reordered_loads, reorder_indices};
  } else {
    return {out_data, recv_sizes};
  }
}

static std::vector<::torch::Tensor> group_asymmetry_all_to_all(const ::torch::Tensor& in_data,
                                                               const ::torch::Tensor& send_sizes,
                                                               bool locality_aware = false) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  const int total_slice_num = send_sizes.numel();
  const int group_size = total_slice_num / world_size;

  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(send_sizes);

  ::torch::Tensor recv_sizes =
      ::torch::empty_like(send_sizes, send_sizes.options(), ::torch::MemoryFormat::Contiguous);
  // at::cuda::CUDACachingAllocator::recordStream(recv_sizes.storage().data_ptr(),
  //                                              at::cuda::getCurrentCUDAStream());

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(send_sizes);
  manager.RecordStorage(recv_sizes);

  distributed::AllToAll(send_sizes.data_ptr(), recv_sizes.data_ptr(),
                        send_sizes.nbytes() / world_size, world_size, manager.GetComm(),
                        manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor all_recv_sizes;
  if (locality_aware) {
    if (world_rank == 0) {
      all_recv_sizes = ::torch::empty({send_sizes.numel() * manager.GetWorldSize()},
                                      send_sizes.options(), ::torch::MemoryFormat::Contiguous);
      // at::cuda::CUDACachingAllocator::recordStream(all_recv_sizes.storage().data_ptr(),
      //                                              at::cuda::getCurrentCUDAStream());
    }
    manager.StartContext();
    manager.WaitEvent(1);
    manager.RecordStorage(recv_sizes);
    if (world_rank == 0) {
      manager.RecordStorage(all_recv_sizes);
      distributed::Gather(recv_sizes.data_ptr(), all_recv_sizes.data_ptr(), recv_sizes.nbytes(), 0,
                          world_rank, world_size, manager.GetComm(), manager.GetStream());
    } else {
      distributed::Gather(recv_sizes.data_ptr(), nullptr, recv_sizes.nbytes(), 0, world_rank,
                          world_size, manager.GetComm(), manager.GetStream());
    }
    manager.RecordEvent(1);
    manager.EndContext();
  }

  auto send_sizes_cpu = send_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());

  // Calculate the total size in byte
  const int total_size_in_byte = in_data.numel() * in_data.element_size();
  // Calculate the size of each grainularity in byte
  const int grain_size_in_byte = total_size_in_byte / in_data.size(0);
  // Calculate the slice and per slice size in byte
  const int slice_size_in_byte = total_size_in_byte / total_slice_num;

  ::torch::Tensor out_data =
      ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);
  // at::cuda::CUDACachingAllocator::recordStream(out_data.storage().data_ptr(),
  //                                              at::cuda::getCurrentCUDAStream());
  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(in_data);
  manager.RecordStorage(out_data);
  distributed::GroupAsymmetryAllToAll(
      in_data.data_ptr(), out_data.data_ptr(), send_sizes_vec, recv_sizes_vec, grain_size_in_byte,
      slice_size_in_byte, group_size, world_size, manager.GetComm(), manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();

  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor reorder_indices;
  ::torch::Tensor all_reordered_loads;
  ::torch::Tensor reordered_loads =
      ::torch::empty_like(send_sizes, send_sizes.options(), ::torch::MemoryFormat::Contiguous);
  // at::cuda::CUDACachingAllocator::recordStream(reordered_loads.storage().data_ptr(),
  //                                              at::cuda::getCurrentCUDAStream());
  if (locality_aware) {
    manager.ExternalWaitEvent(1, at::cuda::getCurrentCUDAStream());
    if (world_rank == 0) {
      auto reorder_results = group_locality_reorder(all_recv_sizes, world_size, group_size);
      all_reordered_loads = reorder_results.first;
      reorder_indices = reorder_results.second;
      // at::cuda::CUDACachingAllocator::recordStream(all_reordered_loads.storage().data_ptr(),
      //                                              at::cuda::getCurrentCUDAStream());
      // at::cuda::CUDACachingAllocator::recordStream(reorder_indices.storage().data_ptr(),
      //                                              at::cuda::getCurrentCUDAStream());
    } else {
      reorder_indices =
          ::torch::empty(world_size, send_sizes.options(), ::torch::MemoryFormat::Contiguous);
      // at::cuda::CUDACachingAllocator::recordStream(reorder_indices.storage().data_ptr(),
      //                                              at::cuda::getCurrentCUDAStream());
    }
    manager.ExternalRecordEvent(1, at::cuda::getCurrentCUDAStream());
    manager.StartContext();
    manager.WaitEvent(1);
    manager.RecordStorage(reorder_indices);
    manager.RecordStorage(reordered_loads);
    if (world_rank == 0) {
      manager.RecordStorage(all_reordered_loads);
      distributed::Scatter(all_reordered_loads.data_ptr(), reordered_loads.data_ptr(),
                           reordered_loads.nbytes(), 0, world_rank, world_size, manager.GetComm(),
                           manager.GetStream());
    } else {
      distributed::Scatter(nullptr, reordered_loads.data_ptr(), reordered_loads.nbytes(), 0,
                           world_rank, world_size, manager.GetComm(), manager.GetStream());
    }

    distributed::BroadCast(reorder_indices.data_ptr(), reorder_indices.data_ptr(),
                           reorder_indices.nbytes(), 0, manager.GetComm(), manager.GetStream());
    manager.RecordEvent(1);
    manager.EndContext();
    manager.ExternalWaitEvent(1, at::cuda::getCurrentCUDAStream());
  }

  if (locality_aware) {
    return {out_data, reordered_loads, reorder_indices};
  } else {
    return {out_data, recv_sizes};
  }
}

static std::tuple<std::vector<::torch::Tensor>, ::torch::Tensor, ::torch::Tensor>
batched_group_asymmetry_all_to_all(const std::vector<::torch::Tensor>& in_datas,
                                   const ::torch::Tensor& send_sizes, bool locality_aware = false) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();
  const int total_slice_num = send_sizes.numel();
  const int group_size = total_slice_num / world_size;
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  CHECK_ON_CUDA(send_sizes);

  ::torch::Tensor recv_sizes =
      ::torch::empty_like(send_sizes, send_sizes.options(), ::torch::MemoryFormat::Contiguous);

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(send_sizes);
  manager.RecordStorage(recv_sizes);

  distributed::AllToAll(send_sizes.data_ptr(), recv_sizes.data_ptr(),
                        send_sizes.nbytes() / world_size, world_size, manager.GetComm(),
                        manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  ::torch::Tensor all_recv_sizes;
  if (locality_aware) {
    if (world_rank == 0) {
      all_recv_sizes = ::torch::empty({send_sizes.numel() * manager.GetWorldSize()},
                                      send_sizes.options(), ::torch::MemoryFormat::Contiguous);
      manager.RecordStorage(all_recv_sizes);
    }
    manager.StartContext();
    manager.WaitEvent(1);
    if (world_rank == 0) {
      distributed::Gather(recv_sizes.data_ptr(), all_recv_sizes.data_ptr(), recv_sizes.nbytes(), 0,
                          world_rank, world_size, manager.GetComm(), manager.GetStream());
    } else {
      distributed::Gather(recv_sizes.data_ptr(), nullptr, recv_sizes.nbytes(), 0, world_rank,
                          world_size, manager.GetComm(), manager.GetStream());
    }
    manager.RecordEvent(1);
    manager.EndContext();
  }

  auto send_sizes_cpu = send_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());

  std::vector<::torch::Tensor> out_datas;
  for (auto& in_data : in_datas) {
    CHECK_ON_CUDA(in_data);

    // Calculate the total size in byte
    const int total_size_in_byte = in_data.numel() * in_data.element_size();
    // Calculate the size of each grainularity in byte
    const int grain_size_in_byte = total_size_in_byte / in_data.size(0);
    // Calculate the slice and per slice size in byte
    const int slice_size_in_byte = total_size_in_byte / total_slice_num;

    ::torch::Tensor out_data =
        ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);
    manager.StartContext();
    manager.WaitEvent(0);
    manager.RecordStorage(in_data);
    manager.RecordStorage(out_data);
    distributed::GroupAsymmetryAllToAll(
        in_data.data_ptr(), out_data.data_ptr(), send_sizes_vec, recv_sizes_vec, grain_size_in_byte,
        slice_size_in_byte, group_size, world_size, manager.GetComm(), manager.GetStream());
    manager.RecordEvent(0);
    manager.EndContext();
    manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());
    out_datas.push_back(out_data);
  }
  ::torch::Tensor reorder_indices;
  ::torch::Tensor all_reordered_loads;
  ::torch::Tensor reordered_loads =
      ::torch::empty_like(send_sizes, send_sizes.options(), ::torch::MemoryFormat::Contiguous);

  if (locality_aware) {
    manager.ExternalWaitEvent(1, at::cuda::getCurrentCUDAStream());
    if (world_rank == 0) {
      auto reorder_results = group_locality_reorder(all_recv_sizes, world_size, group_size);
      all_reordered_loads = reorder_results.first;
      reorder_indices = reorder_results.second;
      manager.RecordStorage(all_reordered_loads);
    } else {
      reorder_indices =
          ::torch::empty(world_size, send_sizes.options(), ::torch::MemoryFormat::Contiguous);
    }
    manager.ExternalRecordEvent(1, at::cuda::getCurrentCUDAStream());
    manager.StartContext();
    manager.WaitEvent(1);
    manager.RecordStorage(reorder_indices);
    manager.RecordStorage(reordered_loads);
    if (world_rank == 0) {
      distributed::Scatter(all_reordered_loads.data_ptr(), reordered_loads.data_ptr(),
                           reordered_loads.nbytes(), 0, world_rank, world_size, manager.GetComm(),
                           manager.GetStream());
    } else {
      distributed::Scatter(nullptr, reordered_loads.data_ptr(), reordered_loads.nbytes(), 0,
                           world_rank, world_size, manager.GetComm(), manager.GetStream());
    }

    distributed::BroadCast(reorder_indices.data_ptr(), reorder_indices.data_ptr(),
                           reorder_indices.nbytes(), 0, manager.GetComm(), manager.GetStream());
    manager.RecordEvent(1);
    manager.EndContext();
    manager.ExternalWaitEvent(1, at::cuda::getCurrentCUDAStream());
  }

  if (locality_aware) {
    return {out_datas, reordered_loads, reorder_indices};
  } else {
    return {out_datas, recv_sizes, recv_sizes};
  }
}

static ::torch::Tensor size_known_group_asymmetry_all_to_all(const ::torch::Tensor& in_data,
                                                             const ::torch::Tensor& send_sizes,
                                                             const ::torch::Tensor& recv_sizes) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  const int total_slice_num = send_sizes.numel();
  const int group_size = total_slice_num / world_size;
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  CHECK_ON_CUDA(in_data);

  auto send_sizes_cpu = send_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());

  // Calculate the total size in byte
  const int total_size_in_byte = in_data.numel() * in_data.element_size();
  // Calculate the size of each grainularity in byte
  const int grain_size_in_byte = total_size_in_byte / in_data.size(0);
  // Calculate the slice and per slice size in byte
  const int slice_size_in_byte = total_size_in_byte / total_slice_num;

  ::torch::Tensor out_data =
      ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);
  // at::cuda::CUDACachingAllocator::recordStream(out_data.storage().data_ptr(),
  //                                              at::cuda::getCurrentCUDAStream());

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(in_data);
  manager.RecordStorage(out_data);
  distributed::GroupAsymmetryAllToAll(
      in_data.data_ptr(), out_data.data_ptr(), send_sizes_vec, recv_sizes_vec, grain_size_in_byte,
      slice_size_in_byte, group_size, world_size, manager.GetComm(), manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  return out_data;
}

static std::vector<::torch::Tensor> batched_size_known_group_asymmetry_all_to_all(
    const std::vector<::torch::Tensor>& in_datas, const ::torch::Tensor& send_sizes,
    const ::torch::Tensor& recv_sizes) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  const int total_slice_num = send_sizes.numel();
  const int group_size = total_slice_num / world_size;
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  auto send_sizes_cpu = send_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());
  std::vector<::torch::Tensor> out_datas;
  for (auto& in_data : in_datas) {
    CHECK_ON_CUDA(in_data);
    // Calculate the total size in byte
    const int total_size_in_byte = in_data.numel() * in_data.element_size();
    // Calculate the size of each grainularity in byte
    const int grain_size_in_byte = total_size_in_byte / in_data.size(0);
    // Calculate the slice and per slice size in byte
    const int slice_size_in_byte = total_size_in_byte / total_slice_num;

    ::torch::Tensor out_data =
        ::torch::empty_like(in_data, in_data.options(), ::torch::MemoryFormat::Contiguous);
    // at::cuda::CUDACachingAllocator::recordStream(out_data.storage().data_ptr(),
    //                                              at::cuda::getCurrentCUDAStream());
    manager.StartContext();
    manager.WaitEvent(0);
    manager.RecordStorage(in_data);
    manager.RecordStorage(out_data);
    distributed::GroupAsymmetryAllToAll(
        in_data.data_ptr(), out_data.data_ptr(), send_sizes_vec, recv_sizes_vec, grain_size_in_byte,
        slice_size_in_byte, group_size, world_size, manager.GetComm(), manager.GetStream());
    manager.RecordEvent(0);
    manager.EndContext();
    manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());
    out_datas.push_back(out_data);
  }
  return out_datas;
}

static std::vector<::torch::Tensor> group_sparse_all_to_all(const ::torch::Tensor& in_data,
                                                            const ::torch::Tensor& send_sizes) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  auto& world_rank = manager.GetWorldRank();
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  const int total_slice_num = send_sizes.numel();
  const int group_size = total_slice_num / world_size;

  CHECK_ON_CUDA(in_data);
  CHECK_ON_CUDA(send_sizes);

  ::torch::Tensor recv_sizes =
      ::torch::empty_like(send_sizes, send_sizes.options(), ::torch::MemoryFormat::Contiguous);

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(send_sizes);
  manager.RecordStorage(recv_sizes);

  distributed::AllToAll(send_sizes.data_ptr(), recv_sizes.data_ptr(),
                        send_sizes.nbytes() / world_size, world_size, manager.GetComm(),
                        manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  recv_sizes = recv_sizes.view({world_size, group_size}).permute({1, 0}).contiguous();

  auto send_sizes_cpu = send_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());

  // const int send_num = send_sizes_cpu.count_nonzero().item<int>();
  // const int recv_num = recv_sizes_cpu.count_nonzero().item<int>();

  // Calculate the total size in byte
  const int total_size_in_byte = in_data.numel() * in_data.element_size();
  // Calculate the size of each grainularity in byte
  const int grain_size_in_byte = total_size_in_byte / in_data.size(0);

  const int recv_num = recv_sizes_cpu.sum().item<int>();

  auto out_data_shape = in_data.sizes().vec();
  out_data_shape[0] = recv_num;

  ::torch::Tensor out_data =
      ::torch::empty(out_data_shape, in_data.options(), ::torch::MemoryFormat::Contiguous);
  // at::cuda::CUDACachingAllocator::recordStream(out_data.storage().data_ptr(),
  //                                              at::cuda::getCurrentCUDAStream());
  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(in_data);
  manager.RecordStorage(out_data);
  distributed::GroupSparseAllToAllForward(in_data.data_ptr(), out_data.data_ptr(), send_sizes_vec,
                                          recv_sizes_vec, grain_size_in_byte, group_size,
                                          world_size, manager.GetComm(), manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();

  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  return {out_data, recv_sizes_cpu, send_sizes_cpu};
}

static ::torch::Tensor size_known_group_sparse_all_to_all(const ::torch::Tensor& in_data,
                                                          const ::torch::Tensor& send_sizes,
                                                          const ::torch::Tensor& recv_sizes) {
  auto& manager = NcclManager::GetManager();
  auto& world_size = manager.GetWorldSize();
  const int total_slice_num = send_sizes.numel();
  const int group_size = total_slice_num / world_size;
  manager.ExternalRecordEvent(0, at::cuda::getCurrentCUDAStream());

  CHECK_ON_CUDA(in_data);

  auto send_sizes_cpu = send_sizes.cpu();
  std::vector<int> send_sizes_vec(send_sizes_cpu.data_ptr<int>(),
                                  send_sizes_cpu.data_ptr<int>() + send_sizes_cpu.numel());
  auto recv_sizes_cpu = recv_sizes.cpu();
  std::vector<int> recv_sizes_vec(recv_sizes_cpu.data_ptr<int>(),
                                  recv_sizes_cpu.data_ptr<int>() + recv_sizes_cpu.numel());

  // Calculate the total size in byte
  const int total_size_in_byte = in_data.numel() * in_data.element_size();
  // Calculate the size of each grainularity in byte
  const int grain_size_in_byte = total_size_in_byte / in_data.size(0);

  const int recv_num = recv_sizes_cpu.sum().item<int>();

  auto out_data_shape = in_data.sizes().vec();
  out_data_shape[0] = recv_num;

  ::torch::Tensor out_data =
      ::torch::empty(out_data_shape, in_data.options(), ::torch::MemoryFormat::Contiguous);

  // at::cuda::CUDACachingAllocator::recordStream(out_data.storage().data_ptr(),
  //                                              at::cuda::getCurrentCUDAStream());

  manager.StartContext();
  manager.WaitEvent(0);
  manager.RecordStorage(in_data);
  manager.RecordStorage(out_data);
  distributed::GroupSparseAllToAllBackward(in_data.data_ptr(), out_data.data_ptr(), send_sizes_vec,
                                           recv_sizes_vec, grain_size_in_byte, group_size,
                                           world_size, manager.GetComm(), manager.GetStream());
  manager.RecordEvent(0);
  manager.EndContext();
  manager.ExternalWaitEvent(0, at::cuda::getCurrentCUDAStream());

  return out_data;
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
  m.def("group_locality_reorder", &brt::backend::torch::group_locality_reorder,
        "group locality reorder", pybind11::arg("loads"), pybind11::arg("wolrd_size"),
        pybind11::arg("group_size") = 1);
  m.def("exchange", &brt::backend::torch::exchange, "Data exchange between ranks",
        pybind11::arg("in_data"), pybind11::arg("reorder_indices"));
  m.def("batched_exchange", &brt::backend::torch::batched_exchange,
        "Batched data exchange between ranks", pybind11::arg("in_datas"),
        pybind11::arg("reorder_indices"));
  m.def("reverse_exchange", &brt::backend::torch::reverse_exchange,
        "Data revrse exchange between ranks", pybind11::arg("in_data"),
        pybind11::arg("reorder_indices"));
  m.def("batched_reverse_exchange", &brt::backend::torch::batched_reverse_exchange,
        "Batched data reverse exchange between ranks", pybind11::arg("in_datas"),
        pybind11::arg("reorder_indices"));
  m.def("asymmetry_all_to_all", &brt::backend::torch::asymmetry_all_to_all, "asymmetry all to all",
        pybind11::arg("in_data"), pybind11::arg("send_sizes"),
        pybind11::arg("locality_aware") = false);
  m.def("group_asymmetry_all_to_all", &brt::backend::torch::group_asymmetry_all_to_all,
        "asymmetry all to all", pybind11::arg("in_data"), pybind11::arg("send_sizes"),
        pybind11::arg("locality_aware") = false);
  m.def("batched_group_asymmetry_all_to_all",
        &brt::backend::torch::batched_group_asymmetry_all_to_all, "asymmetry all to all",
        pybind11::arg("in_datas"), pybind11::arg("send_sizes"),
        pybind11::arg("locality_aware") = false);
  m.def("size_known_group_asymmetry_all_to_all",
        &brt::backend::torch::size_known_group_asymmetry_all_to_all,
        "asymmetry all to all for send sizes and recv size are already known",
        pybind11::arg("in_data"), pybind11::arg("send_sizes"), pybind11::arg("recv_sizes"));
  m.def("batched_size_known_group_asymmetry_all_to_all",
        &brt::backend::torch::batched_size_known_group_asymmetry_all_to_all,
        "batched asymmetry all to all for send sizes and recv size are already known",
        pybind11::arg("in_datas"), pybind11::arg("send_sizes"), pybind11::arg("recv_sizes"));
  m.def("group_sparse_all_to_all", &brt::backend::torch::group_sparse_all_to_all,
        "sparse all to all", pybind11::arg("in_data"), pybind11::arg("send_sizes"));
  m.def("size_known_group_sparse_all_to_all",
        &brt::backend::torch::size_known_group_sparse_all_to_all, "sparse all to all",
        pybind11::arg("in_data"), pybind11::arg("send_sizes"), pybind11::arg("recv_sizes"));
}