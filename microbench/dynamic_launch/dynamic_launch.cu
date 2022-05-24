#include "/brt/runtime/argparse.h"
#include "op.cuh"

#include <assert.h>

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
  // parse arguments including sheduler flags, persistent L2cache
  argparse::ArgumentParser program("dynamic kernel launch test");
  program.add_argument("--device", "-d")
      .help("device id to run on")
      .scan<'i', int>()
      .default_value(0);
  program.add_argument("--iterations", "-i")
      .help("iterations")
      .scan<'i', int>()
      .default_value(1000);
  program.add_argument("--persist", "-p")
      .help("persistent L2cache")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--kernel_id", "-k")
      .help("kernel id")
      .scan<'i', int>()
      .default_value(0);
  program.add_argument("--vector_size", "-v")
      .help("vector size")
      .scan<'i', int>()
      .default_value(1024);
  program.add_argument("--block_size", "-b")
      .help("vector size")
      .scan<'i', int>()
      .default_value(1024);
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // std::cout << program;
  const int device_id = program.get<int>("--device");
  const int test_iterations = program.get<int>("--iterations");
  const bool persistent = program.get<bool>("--persist");
  const int kernel_id = program.get<int>("--kernel_id");
  const int vector_size = program.get<int>("--vector_size");
  const int block_size = program.get<int>("--block_size");

  // set device
  printf("setting to device %d\n", device_id);
  CUDA_CHECK(cudaSetDevice(device_id));
  // create stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int *d_schedule_flag;
  int *h_schedule_flag;
  auto size_schedule_flag = 1024 * sizeof(int);
  CUDA_CHECK(cudaMalloc(&d_schedule_flag, size_schedule_flag));
  CUDA_CHECK(cudaMallocHost(&h_schedule_flag, size_schedule_flag));

  // set stream persistent L2 cache attributes
  if (persistent) {
    printf("trying to set persistent L2 cache\n");
    cudaStreamAttrID stream_attr_id = cudaStreamAttributeAccessPolicyWindow;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t persist_size =
        std::min(int(prop.l2CacheSize * .1), prop.persistingL2CacheMaxSize);
    if (persist_size > 0) {
      printf("setting persistent L2 cache size to %zu bytes\n",
             size_schedule_flag * 2);
      cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                         size_schedule_flag * 2);
      cudaStreamAttrValue stream_attr;
      stream_attr.accessPolicyWindow.base_ptr = d_schedule_flag;
      stream_attr.accessPolicyWindow.num_bytes = size_schedule_flag * 2;
      stream_attr.accessPolicyWindow.hitRatio = 1.0;
      stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      CUDA_CHECK(cudaStreamSetAttribute(stream, stream_attr_id, &stream_attr));
    } else {
      printf("failed to set persistent L2 cache due to its size is 0 bytes\n");
    }
  }

  // create CUDA events
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float elapsed_time;

  // initialize data
  float *d_A, *d_B, *d_C;
  int size_A = vector_size * vector_size, size_B = size_A, size_C = size_A;
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
  int threads_per_block = block_size;
  int blocks_per_grid = size_A / threads_per_block;
  dim3 dim_block(threads_per_block);
  dim3 dim_grid(blocks_per_grid);

  // warm up
  for (int i = 0; i < test_iterations; i++) {
    // set scheduler flag
    set_flag<<<dim_grid, dim_block, 0, stream>>>(d_schedule_flag, kernel_id);
    // compute
    simple_add<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                   d_schedule_flag);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  // reset result data
  printf("testing the lowwer bound");
  SetConstantValue(h_C, size_C, 0.0);
  // test lower bound
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < test_iterations; i++) {
    // set scheduler flag
    set_flag<<<dim_grid, dim_block, 0, stream>>>(d_schedule_flag, kernel_id);
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

  printf("testing cpu dynamic scheduling");
  // reset result data
  SetConstantValue(h_C, size_C, 0.0);
  // test cpu dynamic scheduling
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < test_iterations; i++) {
    // set scheduler flag
    set_flag<<<dim_grid, dim_block, 0, stream>>>(d_schedule_flag, kernel_id);
    CUDA_CHECK(cudaMemcpyAsync(h_schedule_flag, d_schedule_flag,
                               size_schedule_flag, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    switch (*h_schedule_flag) {
      case 0: {
        assert(kernel_id == 0);
        simple_add<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
        break;
      }
      case 1: {
        assert(kernel_id == 1);
        simple_add<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
        break;
      }
      case 2: {
        assert(kernel_id == 2);
        simple_add<<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
        break;
      }
      case 3: {
        assert(kernel_id == 3);
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

  printf("testing gpu dynamic scheduling");
  // reset result data
  SetConstantValue(h_C, size_C, 0.0);
  // test GPU dynamic scheduling
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < test_iterations; i++) {
    // set scheduler flag
    set_flag<<<dim_grid, dim_block, 0, stream>>>(d_schedule_flag, kernel_id);
    dynamic_add_asm<0><<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
    dynamic_add_asm<1><<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
    dynamic_add_asm<2><<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
    dynamic_add_asm<3><<<dim_grid, dim_block, 0, stream>>>(d_A, d_B, d_C, size_A,
                                                       d_schedule_flag);
    cudaGetLastError();
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