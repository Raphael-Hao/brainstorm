/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once
#include <brt/runtime/cuda_utils.h>

namespace brt {
namespace router {

void RouteWithLocalIndices(float* in_data /*[sample_num x sample_dim]*/,
                           float* outdata /*[?load*dst_num x sample_dim]*/,
                           float* gates /*[sample_num x dst_num]*/,
                           int* route_indices /*[sample_num x dst_num]*/,
                           int* dst_loads /*[dst_num]*/, int sample_num, int sample_dim,
                           int dst_num, cudaStream_t stream);

void RouteBackWithLocalIndices(float* in_data /*[?load*dst_num x sample_dim]*/,
                               float* outdata /*[sample_num x sample_dim]*/,
                               float* gates /*[sample_num x dst_num]*/,
                               int* route_indices /*[sample_num x dst_num]*/,
                               int* dst_loads /*[dst_num]*/, int sample_num, int sample_dim,
                               int dst_num, cudaStream_t stream);
}  // namespace router
}  // namespace brt