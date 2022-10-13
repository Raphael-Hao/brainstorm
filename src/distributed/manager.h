/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef SRC_DISTRIBUTED_MANGER_H_
#define SRC_DISTRIBUTED_MANGER_H_

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvrtc.h>
#include <torch/extension.h>

namespace brt {
namespace distributed {
class NCCLManager {
 public:
  static NCCLManager& get_manager();
  void init() {}

 private:
  ncclComm_t comm_;
};
}  // namespace distributed
}  // namespace brt

#endif  // SRC_DISTRIBUTED_MANGER_H_
