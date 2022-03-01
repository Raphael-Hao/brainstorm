/*!
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * \file: /cuda_utils.h
 * \brief:
 * Author: raphael hao
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>

#define CUDA_CHECK(x) __CUDA_CHECK(x, __FILE__, __LINE__)

inline void __CUDA_CHECK(cudaError_t x, const char *file, int line) {
  do {
    if (x != cudaSuccess) {
      fprintf(stderr, "Error: %s, from file <%s>, line %i.\n",
              cudaGetErrorString(x), file, line);
      exit(1);
    }
  } while (0);
}

#define CUBLAS_CHECK(x) __CUBLAS_CHECK(x, __FILE__, __LINE__)

inline void __CUBLAS_CHECK(cublasStatus_t x, const char *file, int line) {
  do {
    if (x != cublasStatus_t::CUBLAS_STATUS_SUCCESS) {
      switch (x) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
          fprintf(
              stderr,
              "cuBLAS Error: CUBLAS_STATUS_NOT_INITIALIZED file: %s line: %d ",
              file, line);
          break;

        case CUBLAS_STATUS_ALLOC_FAILED:
          fprintf(stderr,
                  "cuBLAS Error: CUBLAS_STATUS_ALLOC_FAILED file: %s line: %d ",
                  file, line);
          break;

        case CUBLAS_STATUS_INVALID_VALUE:
          fprintf(
              stderr,
              "cuBLAS Error: CUBLAS_STATUS_INVALID_VALUE file: %s line: %d ",
              file, line);
          break;

        case CUBLAS_STATUS_ARCH_MISMATCH:
          fprintf(
              stderr,
              "cuBLAS Error: CUBLAS_STATUS_ARCH_MISMATCH file: %s line: %d ",
              file, line);
          break;

        case CUBLAS_STATUS_MAPPING_ERROR:
          fprintf(
              stderr,
              "cuBLAS Error: CUBLAS_STATUS_MAPPING_ERROR file: %s line: %d ",
              file, line);
          break;

        case CUBLAS_STATUS_EXECUTION_FAILED:
          fprintf(
              stderr,
              "cuBLAS Error: CUBLAS_STATUS_EXECUTION_FAILED file: %s line: %d ",
              file, line);
          break;

        case CUBLAS_STATUS_INTERNAL_ERROR:
          fprintf(
              stderr,
              "cuBLAS Error: CUBLAS_STATUS_INTERNAL_ERROR file: %s line: %d ",
              file, line);
          break;

        case CUBLAS_STATUS_NOT_SUPPORTED:
          fprintf(
              stderr,
              "cuBLAS Error: CUBLAS_STATUS_NOT_SUPPORTED file: %s line: %d ",
              file, line);
          break;

        case CUBLAS_STATUS_LICENSE_ERROR:
          fprintf(
              stderr,
              "cuBLAS Error: CUBLAS_STATUS_LICENSE_ERROR file: %s line: %d ",
              file, line);
          break;
      }
      exit(1);
    }
  } while (0);
}