/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef INCLUDE_BRT_RUNTIME_CUDA_UTILS_H_
#define INCLUDE_BRT_RUNTIME_CUDA_UTILS_H_

#include <string>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#endif

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#ifdef USE_NCCL
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

#define START_CUDA_TIMER(x, s) \
  cudaEvent_t x##_start;       \
  cudaEvent_t x##_stop;        \
  start_timer(x##_start, x##_stop, s)

#define STOP_CUDA_TIMER(x, s) stop_timer(x##_start, x##_stop, s)

inline void start_timer(cudaEvent_t& start, cudaEvent_t& stop, cudaStream_t stream = 0) {
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
}

inline void stop_timer(cudaEvent_t& start, cudaEvent_t& stop, cudaStream_t stream = 0) {
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_time;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("Elapsed time: %f ms\n", elapsed_time);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
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

#if defined(USE_NCCL)
#define NCCL_CHECK(x) __NCCL_CHECK(x, __FILE__, __LINE__)

inline void __NCCL_CHECK(ncclResult_t x, const char* file, int line) {
  do {
    if (x != ncclResult_t::ncclSuccess) {
      switch (x) {
        case ncclResult_t::ncclUnhandledCudaError:
          fprintf(stderr, "NCCL Error: ncclUnhandledCudaError, file: %s line: %d ", file, line);
          break;

        case ncclResult_t::ncclSystemError:
          fprintf(stderr, "NCCL Error: ncclSystemError, file: %s line: %d ", file, line);
          break;

        case ncclResult_t::ncclInternalError:
          fprintf(stderr, "NCCL Error: ncclInternalError, file: %s line: %d ", file, line);
          break;

        case ncclResult_t::ncclInvalidArgument:
          fprintf(stderr, "NCCL Error: ncclInvalidArgument, file: %s line: %d ", file, line);
          break;

        case ncclResult_t::ncclInvalidUsage:
          fprintf(stderr, "NCCL Error: ncclInvalidUsage, file: %s line: %d ", file, line);
          break;

        case ncclResult_t::ncclNumResults:
          fprintf(stderr, "NCCL Error: ncclNumResults, file: %s line: %d ", file, line);
          break;
        default:
          fprintf(stderr, "NCCL Error: Unknown error, file: %s line: %d ", file, line);
          break;
      }
      exit(1);
    }
  } while (0);
}

#endif

}  // namespace brt

#endif  // INCLUDE_BRT_RUNTIME_CUDA_UTILS_H_
