/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_DISTRIBUTED_MANAGER_H_
#define BRT_DISTRIBUTED_MANAGER_H_

#include <brt/runtime/cuda_utils.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "./torch.h"

namespace brt {
namespace backend {
namespace torch {
class NcclManager {
 public:
  static NcclManager& GetManager();

  void Init(const ::torch::Tensor& unique_id_t, const int& world_rank, const int& world_size);
  // context
  void StartContext();
  void EndContext();
  // memory
  void RecordStorage(const ::torch::Tensor& T);
  // synchorization
  void RecordEvent(const int& event_id);
  void WaitEvent(const int& event_id);
  void ExternalRecordEvent(const int& event_id, const at::cuda::CUDAStream& ext_stream);
  void ExternalWaitEvent(const int& event_id, const at::cuda::CUDAStream& ext_stream);

  ncclComm_t GetComm() { return comm_; }
  cudaStream_t GetStream() { return stream_; }
  const int& GetWorldRank() { return world_rank_; }
  const int& GetWorldSize() { return world_size_; }

  bool IsInitialized() { return initialized_; }

 private:
  NcclManager() : stream_(at::cuda::getStreamFromPool()), original_stream_(c10::nullopt) {
    initialized_ = false;
  }
  ~NcclManager() { NCCL_CHECK(ncclCommDestroy(comm_)); }
  bool initialized_;
  int world_rank_;
  int world_size_;
  ncclComm_t comm_;
  at::cuda::CUDAStream stream_;
  c10::optional<at::cuda::CUDAStream> original_stream_;
  std::vector<at::cuda::CUDAEvent> events_;
};
}  // namespace torch
}  // namespace backend
}  // namespace brt

#endif  // BRT_DISTRIBUTED_MANAGER_H_
