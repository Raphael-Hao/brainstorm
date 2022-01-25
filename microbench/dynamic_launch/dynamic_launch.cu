#include "op.cuh"

void SetConstantValue(float *h_data, int size, float value) {
  for (int i = 0; i < size; i++) {
    h_data[i] = value;
  }
}

void CheckResults(float *h_data, int size, float value, std::string name) {
  for (int i = 0; i < size; i++) {
    if (h_data[i] != value) {
      printf("%s test failed, Error: %d-th data %f != %f\n", name.c_str(), i,
             h_data[i], value);
      exit(1);
    }
  }
}

int main(int argc, char **argv) {
  CUDA_CHECK(cudaSetDevice(0));
  const int test_iterations = 1;
  helloFromGpu<<<1, 10>>>();
  CUDA_CHECK(cudaDeviceSynchronize());

  // create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int *d_schedule_flag;
  int *h_schedule_flag;
  CUDA_CHECK(cudaMalloc(&d_schedule_flag, 16 * sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&h_schedule_flag, 16 * sizeof(int)));

  // set stream persistent L2 cache attributes
  cudaStreamAttrID stream_attr_id = cudaStreamAttributeAccessPolicyWindow;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  size_t persist_size =
      min(int(prop.l2CacheSize * .1), prop.persistingL2CacheMaxSize);
  if (persist_size > 0) {
    printf("setting persistent L2 cache size to %zu bytes\n", persist_size);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size);
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = d_schedule_flag;
    stream_attr.accessPolicyWindow.num_bytes = 16 * sizeof(int);
    stream_attr.accessPolicyWindow.hitRatio = 1.0;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    CUDA_CHECK(cudaStreamSetAttribute(stream, stream_attr_id, &stream_attr));
  } else {
    printf("persistent L2 cache size is 0 bytes\n");
  }
  // create CUDA events
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float elapsed_time;

  // initialize data
  float *d_A, *d_B, *d_C;
  int size_A = 1024 * 1024, size_B = size_A, size_C = size_A;
  CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
  float *h_A, *h_B, *h_C;
  CUDA_CHECK(cudaMallocHost(&h_A, size_A * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_B, size_B * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_C, size_C * sizeof(float)));
  SetConstantValue(h_A, size_A, 1.0);
  SetConstantValue(h_B, size_B, 1.0);
  CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, size_A * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, size_B * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CheckResults(h_A, size_A, 1.0, "A");
  CheckResults(h_B, size_B, 1.0, "B");

  // create grid and block
  int threads_per_block = 1024;
  int blocks_per_grid = size_A / threads_per_block;
  dim3 dim_block(threads_per_block);
  dim3 dim_grid(blocks_per_grid);

  // reset result data
  SetConstantValue(h_C, size_C, 0.0);
  // test lower bound
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < test_iterations; i++) {
    // set scheduler flag
    CUDA_CHECK(cudaMemsetAsync(d_schedule_flag, 0, 16 * sizeof(int), stream));
    // compute
    simple_add<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                   d_schedule_flag);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, size_C * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CheckResults(h_C, size_C, 2.0, "lower bound");
  printf("lower bound: %f ms\n", elapsed_time / test_iterations);

  // reset result data
  SetConstantValue(h_C, size_C, 0.0);
  // test cpu dynamic scheduling
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < test_iterations; i++) {
    // set scheduler flag
    CUDA_CHECK(cudaMemsetAsync(d_schedule_flag, 0, 16 * sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(h_schedule_flag, d_schedule_flag, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    switch (*h_schedule_flag) {
      case 0:
      case 1:
      case 2:
      case 3: {
        simple_add<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
        break;
      }
      default:
        break;
    }
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, size_C * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CheckResults(h_C, size_C, 2.0, "cpu dynamic scheduling");
  printf("cpu dynamic scheduling: %f ms\n", elapsed_time / test_iterations);

  // reset result data
  SetConstantValue(h_C, size_C, 0.0);
  // test GPU dynamic scheduling
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < test_iterations; i++) {
    // set scheduler flag
    CUDA_CHECK(cudaMemsetAsync(d_schedule_flag, 0, 16 * sizeof(int), stream));
    dynamic_add<0><<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
    dynamic_add<1><<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
    dynamic_add<2><<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
    dynamic_add<3><<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
  CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, size_C * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CheckResults(h_C, size_C, 2.0, "gpu dynamic scheduling");
  printf("gpu dynamic scheduling: %f ms\n", elapsed_time / test_iterations);
}