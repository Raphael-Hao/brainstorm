/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/distributed/collective.h>
#include <brt/runtime/cuda_utils.h>

namespace brt {

namespace distributed {

void Gather(const void* sendbuf, void* recvbuf, const int& send_size_in_byte, const int& root,
            const int& world_rank, const int& world_size, ncclComm_t comm, cudaStream_t stream) {
  NCCL_CHECK(ncclGroupStart());
  if (world_rank == root) {
    for (int i = 0; i < world_size; i++) {
      NCCL_CHECK(ncclRecv((char*)recvbuf + i * send_size_in_byte, send_size_in_byte, ncclChar, i, comm, stream));
    }
  }
  NCCL_CHECK(ncclSend(sendbuf, send_size_in_byte, ncclChar, root, comm, stream));
  NCCL_CHECK(ncclGroupEnd());
}

void AllToAll(void* send_buffer, void* recv_buffer, const int& slice_size_in_byte,
              const int& slice_num, ncclComm_t comm, cudaStream_t stream) {
  NCCL_CHECK(ncclGroupStart());
  for (auto i = 0; i < slice_num; i++) {
    NCCL_CHECK(ncclSend((char*)send_buffer + i * slice_size_in_byte, slice_size_in_byte, ncclChar,
                        i, comm, stream));
    NCCL_CHECK(ncclRecv((char*)recv_buffer + i * slice_size_in_byte, slice_size_in_byte, ncclChar,
                        i, comm, stream));
  }
  NCCL_CHECK(ncclGroupEnd());
}

void AsymmetryAllToAll(void* send_buffer, void* recv_buffer, const std::vector<int>& send_sizes,
                       const std::vector<int>& recv_sizes, const int& grain_size_in_byte,
                       const int& slice_size_in_byte, const int& slice_num, ncclComm_t comm,
                       cudaStream_t stream) {
  NCCL_CHECK(ncclGroupStart());
  for (auto i = 0; i < slice_num; i++) {
    NCCL_CHECK(ncclSend((char*)send_buffer + i * slice_size_in_byte,
                        send_sizes[i] * grain_size_in_byte, ncclChar, i, comm, stream));
    NCCL_CHECK(ncclRecv((char*)recv_buffer + i * slice_size_in_byte,
                        recv_sizes[i] * grain_size_in_byte, ncclChar, i, comm, stream));
  }
  NCCL_CHECK(ncclGroupEnd());
}

}  // namespace distributed

}  // namespace brt
