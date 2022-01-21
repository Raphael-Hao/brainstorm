#include "op.cuh"

int main(int argc, char **argv) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  size_t persist_size =
      min(int(prop.l2CacheSize * .5), prop.persistingL2CacheMaxSize);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 128 * 1024);
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
}