/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#pragma once
#include <brt/runtime/cuda_utils.h>

namespace brt {
namespace netlet {

template <typename T>
void DevicePtr2PtrArray(T** dst, T* src, int index[], int array_size, int granularity,
                        cudaStream_t stream);

}
}  // namespace brt