/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef SRC_DISTRIBUTED_MANGER_H_
#define SRC_DISTRIBUTED_MANGER_H_

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

  void init(ncclComm_t comm, cudaStream_t stream) {
    set_stream(stream);
    set_comm(comm);
  }

  void set_stream(cudaStream_t stream) { stream_ = stream; }
  void set_comm(ncclComm_t comm) { comm_ = comm; }

  ncclComm_t get_nccl_comm() { return comm_; }
  cudaStream_t get_nccl_stream() { return stream_; }

 private:
  ncclComm_t comm_;
  cudaStream_t stream_;
};
}  // namespace distributed
}  // namespace brt

#endif  // SRC_DISTRIBUTED_MANGER_H_
