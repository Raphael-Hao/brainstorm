/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_RUNTIME_CUDA_UTILS_H_
#define BRT_RUNTIME_CUDA_UTILS_H_

#include <string>
// cuda
#if defined(USE_CUDA)

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#endif

// cublas
#if defined(USE_CUBLAS)
#include <cublas_v2.h>
#endif

// cudnn
#if defined(USE_CUDNN)
#include <cudnn.h>
#endif

// nccl
#if defined(USE_NCCL)
#include <nccl.h>
#endif

namespace brt {

#if defined(USE_CUDA)

#define CUDA_CHECK(x) __CUDA_CHECK(x, __FILE__, __LINE__)

inline void __CUDA_CHECK(cudaError_t x, const char* file, int line) {
  do {
    if (x != cudaSuccess) {
      fprintf(stderr, "Error: %s, from file <%s>, line %i.\n", cudaGetErrorString(x), file, line);
      exit(1);
    }
  } while (0);
}

#define CU_CHECK(x) __CU_CHECK(x, __FILE__, __LINE__)

inline void __CU_CHECK(CUresult x, const char* file, int line) {
  do {
    if (x != cudaError_enum::CUDA_SUCCESS) {
      const char* error_str;
      cuGetErrorString(x, &error_str);
      fprintf(stderr, "Error: %s, from file <%s>, line %i.\n", error_str, file, line);
      exit(1);
    }
  } while (0);
}

#define NVRTC_CHECK(x) __NVRTC_CHECK(x, __FILE__, __LINE__)

inline void __NVRTC_CHECK(nvrtcResult x, const char* file, int line) {
  do {
    if (x != nvrtcResult::NVRTC_SUCCESS) {
      fprintf(stderr, "Error: %s, from file <%s>, line %i.\n", nvrtcGetErrorString(x), file, line);
      exit(1);
    }
  } while (0);
}

#endif

#if defined(USE_CUBLAS)

#define CUBLAS_CHECK(x) __CUBLAS_CHECK(x, __FILE__, __LINE__)

inline void __CUBLAS_CHECK(cublasStatus_t x, const char* file, int line) {
  do {
    if (x != cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
      switch (x) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_NOT_INITIALIZED file: %s line: %d ", file,
                  line);
          break;

        case CUBLAS_STATUS_ALLOC_FAILED:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_ALLOC_FAILED file: %s line: %d ", file,
                  line);
          break;

        case CUBLAS_STATUS_INVALID_VALUE:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_INVALID_VALUE file: %s line: %d ", file,
                  line);
          break;

        case CUBLAS_STATUS_ARCH_MISMATCH:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_ARCH_MISMATCH file: %s line: %d ", file,
                  line);
          break;

        case CUBLAS_STATUS_MAPPING_ERROR:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_MAPPING_ERROR file: %s line: %d ", file,
                  line);
          break;

        case CUBLAS_STATUS_EXECUTION_FAILED:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_EXECUTION_FAILED file: %s line: %d ", file,
                  line);
          break;

        case CUBLAS_STATUS_INTERNAL_ERROR:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_INTERNAL_ERROR file: %s line: %d ", file,
                  line);
          break;

        case CUBLAS_STATUS_NOT_SUPPORTED:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_NOT_SUPPORTED file: %s line: %d ", file,
                  line);
          break;

        case CUBLAS_STATUS_LICENSE_ERROR:
          fprintf(stderr, "cuBLAS Error: CUBLAS_STATUS_LICENSE_ERROR file: %s line: %d ", file,
                  line);
          break;
      }
      exit(1);
    }
  } while (0);
}
#endif

}  // namespace brt

#endif  // BRT_RUNTIME_CUDA_UTILS_H_
