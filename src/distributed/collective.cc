/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/distributed/collective.h>
#include <brt/runtime/cuda_utils.h>

namespace brt {

namespace distributed {

void BroadCast(void* send_buffer, void* recv_buffer, const int& send_size_in_byte, const int& root,
               ncclComm_t comm, cudaStream_t stream) {
  NCCL_CHECK(
      ncclBroadcast(send_buffer, recv_buffer, send_size_in_byte, ncclChar, root, comm, stream));
}

void Scatter(void* send_buffer, void* recv_buffer, const int& send_size_in_byte, const int& root,
             const int& world_rank, const int& world_size, ncclComm_t comm, cudaStream_t stream) {
  NCCL_CHECK(ncclGroupStart());
  if (world_rank == root) {
    for (int i = 0; i < world_size; i++) {
      NCCL_CHECK(ncclSend((char*)send_buffer + i * send_size_in_byte, send_size_in_byte, ncclChar,
                          i, comm, stream));
    }
  }
  NCCL_CHECK(ncclRecv(recv_buffer, send_size_in_byte, ncclChar, root, comm, stream));
  NCCL_CHECK(ncclGroupEnd());
}

void Gather(void* send_buffer, void* recv_buffer, const int& send_size_in_byte, const int& root,
            const int& world_rank, const int& world_size, ncclComm_t comm, cudaStream_t stream) {
  NCCL_CHECK(ncclGroupStart());
  if (world_rank == root) {
    for (int i = 0; i < world_size; i++) {
      NCCL_CHECK(ncclRecv((char*)recv_buffer + i * send_size_in_byte, send_size_in_byte, ncclChar,
                          i, comm, stream));
    }
  }
  NCCL_CHECK(ncclSend(send_buffer, send_size_in_byte, ncclChar, root, comm, stream));
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
void GroupAsymmetryAllToAll(void* send_buffer, void* recv_buffer,
                            const std::vector<int>& send_sizes, const std::vector<int>& recv_sizes,
                            const int& grain_size_in_byte, const int& slice_size_in_byte,
                            const int& group_size, const int& world_size, ncclComm_t comm,
                            cudaStream_t stream) {
  const int group_size_in_byte = group_size * slice_size_in_byte;
  int group_base_idx = 0;
  NCCL_CHECK(ncclGroupStart());
  for (auto i = 0; i < world_size; i++) {
    for (auto j = 0; j < group_size; j++) {
      NCCL_CHECK(ncclSend((char*)send_buffer + j * slice_size_in_byte,
                          send_sizes[group_base_idx + j] * grain_size_in_byte, ncclChar, i, comm,
                          stream));
      NCCL_CHECK(ncclRecv((char*)recv_buffer + j * slice_size_in_byte,
                          recv_sizes[group_base_idx + j] * grain_size_in_byte, ncclChar, i, comm,
                          stream));
    }
    send_buffer = (char*)send_buffer + group_size_in_byte;
    recv_buffer = (char*)recv_buffer + group_size_in_byte;
    group_base_idx += group_size;
  }
  NCCL_CHECK(ncclGroupEnd());
}

}  // namespace distributed

}  // namespace brt
