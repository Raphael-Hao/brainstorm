#include "op.cuh"

int main(int argc, char **argv) {
  // create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  // set stream persistent L2 cache attributes
  cudaStreamAttrID stream_attr_id = cudaStreamAttributeAccessPolicyWindow;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  size_t persist_size =
      min(int(prop.l2CacheSize * .1), prop.persistingL2CacheMaxSize);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size);
  cudaStreamAttrValue stream_attr;
  int *schedule_flag;
  CUDA_CHECK(cudaMalloc(&schedule_flag, 16 * sizeof(int)));
  stream_attr.accessPolicyWindow.base_ptr = schedule_flag;
  stream_attr.accessPolicyWindow.num_bytes = 16 * sizeof(int);
  stream_attr.accessPolicyWindow.hitRatio = 1.0;
  stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  CUDA_CHECK(cudaStreamSetAttribute(stream, stream_attr_id, &stream_attr));

  float *d_A, *d_B, *d_C;
  int size_A = 1024 * 1024, size_B = size_A, size_C = size_A;
  CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));

  // create grid and block
  int block_size = 32;
  dim3 threads = dim3(block_size, block_size);
  dim3 grid(size_A / (threads.x * threads.y), 1);

  // set scheduler flag
  CUDA_CHECK(cudaMemsetAsync(schedule_flag, 0, 16 * sizeof(int), stream));

  // launch candidate kernels
  kernel_add<0>
      <<<grid, threads, 0, stream>>>(d_A, d_B, d_C, size_A, schedule_flag);
  kernel_add<1>
      <<<grid, threads, 0, stream>>>(d_A, d_B, d_C, size_A, schedule_flag);
  kernel_add<2>
      <<<grid, threads, 0, stream>>>(d_A, d_B, d_C, size_A, schedule_flag);
  kernel_add<3>
      <<<grid, threads, 0, stream>>>(d_A, d_B, d_C, size_A, schedule_flag);
  CUDA_CHECK(cudaStreamSynchronize(stream));

}