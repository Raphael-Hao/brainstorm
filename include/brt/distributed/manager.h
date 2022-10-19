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
#include <nvrtc.h>

namespace brt {
namespace distributed {
class NcclManager {
 public:
  static NcclManager& GetManager();

  void Init(void* unique_id_ptr, const int& world_rank, const int& world_size,
            cudaStream_t stream) {
    SetStream(stream);
    InitComm(unique_id_ptr, world_rank, world_size);
    initialized_ = true;
  }

  int GetNcclUniqueIDSize() { return sizeof(ncclUniqueId); }

  ncclComm_t GetNcclComm() { return comm_; }
  cudaStream_t GetNcclStream() { return stream_; }
  bool IsInitialized() { return initialized_; }

 private:
  void SetStream(cudaStream_t stream) { stream_ = stream; }
  void InitComm(void* unique_id_ptr, int world_rank, int world_size) {
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclCommInitRank(&comm_, world_size, *(ncclUniqueId*)unique_id_ptr, world_rank));
    NCCL_CHECK(ncclGroupEnd());
  }
  NcclManager() { initialized_ = false; }
  bool initialized_;
  ncclComm_t comm_;
  cudaStream_t stream_;
};
}  // namespace distributed
}  // namespace brt

#endif  // BRT_DISTRIBUTED_MANAGER_H_
