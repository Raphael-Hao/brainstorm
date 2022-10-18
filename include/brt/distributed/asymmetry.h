/*!
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * \file: /asymmetry.h
 * \brief:
 * Author: raphael hao
 */

#ifndef BRT_DISTRIBUTED_ASYMMETRY_H_
#define BRT_DISTRIBUTED_ASYMMETRY_H_

#include <vector>

namespace brt {
namespace distributed {
void AsymmetryAllToAll(void* send_buffer, void* recv_buffer, const std::vector<int>& send_sizes,
                       const std::vector<int>& recv_sizes, const int& grain_size_in_byte,
                       const int& slice_size_in_byte, const int& slice_num);
}
}  // namespace brt
#endif  // BRT_DISTRIBUTED_ASYMMETRY_H_
