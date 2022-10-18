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
class NCCLManager {
 public:
  static NCCLManager& get_manager();

  void init(void* unique_id_ptr, const int& world_rank, const int& world_size,
            cudaStream_t stream) {
    set_stream(stream);
    init_comm(unique_id_ptr, world_rank, world_size);
  }

  void set_stream(cudaStream_t stream) { stream_ = stream; }
  void init_comm(void* unique_id_ptr, int world_rank, int world_size) {
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclCommInitRank(&comm_, world_size, *(ncclUniqueId*)unique_id_ptr, world_rank));
    NCCL_CHECK(ncclGroupEnd());
  }
  int get_nccl_unique_id_size() { return sizeof(ncclUniqueId); }

  ncclComm_t get_nccl_comm() { return comm_; }
  cudaStream_t get_nccl_stream() { return stream_; }

 private:
  ncclComm_t comm_;
  cudaStream_t stream_;
};
}  // namespace distributed
}  // namespace brt

#endif  // BRT_DISTRIBUTED_MANAGER_H_
