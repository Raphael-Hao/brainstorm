/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include "./nccl_manager.h"

#include <c10/cuda/CUDACachingAllocator.h>

namespace brt {
namespace backend {
namespace torch {

NcclManager& NcclManager::GetManager() {
  static NcclManager manager;
  return manager;
}

void NcclManager::Init(const ::torch::Tensor& unique_id_t, const int& world_rank,
                       const int& world_size) {
  world_rank_ = world_rank;
  world_size_ = world_size;
  ncclUniqueId nccl_unique_id;
  memcpy(&nccl_unique_id, (void*)unique_id_t.data_ptr(), sizeof(ncclUniqueId));
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclCommInitRank(&comm_, world_size, nccl_unique_id, world_rank));
  NCCL_CHECK(ncclGroupEnd());
  events_.resize(1);
  initialized_ = true;
}

void NcclManager::StartContext() {
  original_stream_ = at::cuda::getCurrentCUDAStream();
  at::cuda::setCurrentCUDAStream(stream_);
}

void NcclManager::EndContext() { at::cuda::setCurrentCUDAStream(*original_stream_); }
void NcclManager::RecordStorage(const ::torch::Tensor& t) {
  at::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), stream_);
}
void NcclManager::RecordEvent(const int& event_id) {
  if (event_id >= events_.size()) {
    events_.resize(event_id + 1);
  }
  events_[event_id].record(stream_);
}
void NcclManager::WaitEvent(const int& event_id) {
  CHECK_LT(event_id, events_.size());
  events_[event_id].block(stream_);
}
void NcclManager::ExternalRecordEvent(const int& event_id, const at::cuda::CUDAStream& stream) {
  if (event_id >= events_.size()) {
    events_.resize(event_id + 1);
  }
  events_[event_id].record(stream);
}
void NcclManager::ExternalWaitEvent(const int& event_id, const at::cuda::CUDAStream& stream) {
  CHECK_LT(event_id, events_.size());
  events_[event_id].block(stream);
}

}  // namespace torch
}  // namespace backend
}  // namespace brt
