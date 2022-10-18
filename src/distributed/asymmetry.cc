/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include <brt/runtime/cuda_utils.h>

#include <vector>

#include "./manager.h"

namespace brt {

namespace distributed {
void AsymmetryAllToAll(void* send_buffer, void* recv_buffer, const std::vector<int>& send_sizes,
                       const std::vector<int>& recv_sizes, const int& slice_size_in_byte,
                       const int& slice_num) {
  NCCLManager& manager = NCCLManager::get_manager();
  NCCL_CHECK(ncclGroupStart());
  for (auto i = 0; i < slice_num; i++) {
    NCCL_CHECK(ncclSend((char*)send_buffer + i * slice_size_in_byte, send_sizes[i], ncclInt8, i,
                        manager.get_nccl_comm(), manager.get_nccl_stream()));
    NCCL_CHECK(ncclRecv((char*)recv_buffer + i * slice_size_in_byte, recv_sizes[i], ncclInt8, i,
                        manager.get_nccl_comm(), manager.get_nccl_stream()));
  }
  NCCL_CHECK(ncclGroupEnd());
}

}  // namespace distributed

}  // namespace brt
