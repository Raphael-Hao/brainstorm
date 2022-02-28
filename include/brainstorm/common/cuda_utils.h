/*!
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * \file: /cuda_utils.h
 * \brief:
 * Author: raphael hao
 */

#include <cuda_runtime.h>
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