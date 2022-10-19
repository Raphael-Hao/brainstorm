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

  void Init(::torch::Tensor unique_id_t, const int& world_rank, const int& world_size,
            const int& event_num = 1);
  void StartContext();
  void EndContext();
  void RecordStorage(const ::torch::Tensor& T);
  void RecordEvent(const int& event_id);
  void WaitEvent(const int& event_id);
  void ExternalRecordEvent(const int& event_id, at::cuda::CUDAStream stream);
  void ExternalWaitEvent(const int& event_id, at::cuda::CUDAStream stream);
  int GetNcclUniqueIDSize() { return sizeof(ncclUniqueId); }

  ncclComm_t GetComm() { return comm_; }
  cudaStream_t GetStream() { return stream_; }
  bool IsInitialized() { return initialized_; }

 private:
  ~NcclManager() { NCCL_CHECK(ncclCommDestroy(comm_)); }
  NcclManager()
      : stream_(at::cuda::getStreamFromPool()), original_stream_(at::cuda::getCurrentCUDAStream()) {
    initialized_ = false;
  }
  bool initialized_;
  ncclComm_t comm_;
  at::cuda::CUDAStream stream_;
  at::cuda::CUDAStream original_stream_;
  std::vector<at::cuda::CUDAEvent> events_;
};
}  // namespace torch
}  // namespace backend
}  // namespace brt

#endif  // BRT_DISTRIBUTED_MANAGER_H_
