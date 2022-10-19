/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/distributed/asymmetry.h>
#include <brt/runtime/cuda_utils.h>

namespace brt {

namespace distributed {
void AsymmetryAllToAll(void* send_buffer, void* recv_buffer, const std::vector<int>& send_sizes,
                       const std::vector<int>& recv_sizes, const int& grain_size_in_byte,
                       const int& slice_size_in_byte, const int& slice_num, ncclComm_t comm,
                       cudaStream_t stream) {
  NCCL_CHECK(ncclGroupStart());
  for (auto i = 0; i < slice_num; i++) {
    NCCL_CHECK(ncclSend((char*)send_buffer + i * slice_size_in_byte,
                        send_sizes[i] * grain_size_in_byte, ncclInt8, i, comm, stream));
    NCCL_CHECK(ncclRecv((char*)recv_buffer + i * slice_size_in_byte,
                        recv_sizes[i] * grain_size_in_byte, ncclInt8, i, comm, stream));
  }
  NCCL_CHECK(ncclGroupEnd());
}

}  // namespace distributed

}  // namespace brt
