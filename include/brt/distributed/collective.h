/*!
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * \file: /asymmetry.h
 * \brief:
 * Author: raphael hao
 */

#ifndef BRT_DISTRIBUTED_COLLECTIVE_H_
#define BRT_DISTRIBUTED_COLLECTIVE_H_

#include <cuda_runtime.h>
#include <nccl.h>

#include <vector>

namespace brt {
namespace distributed {
void BroadCast(void* send_buffer, void* recv_buffer, const int& send_size_in_byte, const int& root,
               ncclComm_t comm, cudaStream_t stream);
void Scatter(void* send_buffer, void* recv_buffer, const int& send_size_in_byte, const int& root,
             const int& world_rank, const int& world_size, ncclComm_t comm, cudaStream_t stream);
void Gather(void* sendbuf, void* recvbuf, const int& send_size_in_byte, const int& root,
            const int& world_rank, const int& world_size, ncclComm_t comm, cudaStream_t stream);
void AllToAll(void* send_buffer, void* recv_buffer, const int& slice_size_in_byte,
              const int& slice_num, ncclComm_t comm, cudaStream_t stream);
void AsymmetryAllToAll(void* send_buffer, void* recv_buffer, const std::vector<int>& send_sizes,
                       const std::vector<int>& recv_sizes, const int& grain_size_in_byte,
                       const int& slice_size_in_byte, const int& slice_num, ncclComm_t comm,
                       cudaStream_t stream);
}  // namespace distributed
}  // namespace brt
#endif  // BRT_DISTRIBUTED_COLLECTIVE_H_
