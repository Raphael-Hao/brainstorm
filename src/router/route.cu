
#include <brt/router/route.h>

namespace brt {
namespace router {

__global__ void __launch_bounds__(1024) dispatch_with_dst_indices_2d(
    float* __restrict__ in_data /*[path_num x sample_num x sample_dim]*/,
    float* __restrict__ out_data /*[?load*path_num x sample_dim]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/, int sample_num, int sample_dim, int path_num) {
  in_data += sample_num * sample_dim * blockIdx.x;

  int load_start = 0;
  for (int i = 0; i < blockIdx.y; i++) {
    load_start += loads[i];
  }

  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst = route_indices[route_index];
    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }
    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < sample_dim; j += 1024) {
      out_data[global_dst * sample_dim + j] = in_data[i * sample_dim + j];
    }
  }
}

__global__ void __launch_bounds__(1024) no_transform_dispatch_with_dst_indices(
    float* __restrict__ in_data /*[sample_num x sample_dim]*/,
    float* __restrict__ out_data /*[?load*path_num x sample_dim]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/, int sample_num, int sample_dim, int path_num) {
  int load_start = 0;
  for (int i = 0; i < blockIdx.y; i++) {
    load_start += loads[i];
  }

  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst = route_indices[route_index];
    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }
    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < sample_dim; j += 1024) {
      out_data[global_dst * sample_dim + j] = in_data[i * sample_dim + j];
    }
  }
}

__global__ void __launch_bounds__(1024)
    dipatch_with_dst_indices(float* __restrict__ in_data /*[sample_num x sample_dim]*/,
                             float* __restrict__ out_data /*[?load*path_num x sample_dim]*/,
                             float* __restrict__ gates /*[sample_num x path_num]*/,
                             int* __restrict__ route_indices /*[sample_num x path_num]*/,
                             int* __restrict__ loads /*[path_num]*/, int sample_num, int sample_dim,
                             int path_num) {
  int load_start = 0;
  for (int i = 0; i < blockIdx.y; i++) {
    load_start += loads[i];
  }

  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;

    int local_dst = route_indices[route_index];
    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }
    int global_dst = local_dst - 1 + load_start;

    for (int j = threadIdx.x; j < sample_dim; j += 1024) {
      out_data[global_dst * sample_dim + j] = in_data[i * sample_dim + j] * gates[route_index];
    }
  }
}

__global__ void __launch_bounds__(1024) no_transform_combine_with_src_indices(
    float* __restrict__ in_data /*[?load*path_num x sample_dim]*/,
    float* __restrict__ out_data /*[sample_num x sample_dim]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/, int sample_num, int sample_dim, int path_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];
      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }
      int global_dst = local_dst - 1;
      for (int k = 0; k < j; k++) {
        global_dst += loads[k];
      }
      for (int k = threadIdx.x; k < sample_dim; k += 1024) {
        out_data[i * sample_dim + k] += in_data[global_dst * sample_dim + k];
      }
    }
  }
}

__global__ void __launch_bounds__(1024)
    combine_with_src_indices(float* __restrict__ in_data /*[?load*path_num x sample_dim]*/,
                             float* __restrict__ out_data /*[sample_num x sample_dim]*/,
                             float* __restrict__ gates /*[sample_num x path_num]*/,
                             int* __restrict__ route_indices /*[sample_num x path_num]*/,
                             int* __restrict__ loads /*[path_num]*/, int sample_num, int sample_dim,
                             int path_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];
      if (local_dst == 0) {
        continue;
      }
      int global_dst = local_dst - 1;
      for (int k = 0; k < j; k++) {
        global_dst += loads[k];
      }
      for (int k = threadIdx.x; k < sample_dim; k += 1024) {
        out_data[i * sample_dim + k] += in_data[global_dst * sample_dim + k] * gates[route_index];
      }
    }
  }
}

__global__ void __launch_bounds__(1024) no_transform_route_with_in_indices(
    float* __restrict__ in_data /*[sample_num x sample_dim]*/,
    float* __restrict__ out_data /*[?load*path_num x sample_dim]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ path_loads /*[path_num]*/, int* __restrict__ capacities /*[path_num] */,
    int sample_dim, int path_num) {
  // [thread_extent] blockIdx.x = capacities max
  // [thread_extent] blockIdx.y = 4
  // [thread_extent] threadIdx.x = 1024
  for (int path_idx = blockIdx.y; path_idx < path_num; path_idx += gridDim.y) {
    int base_index = 0;
    for (int i = 0; i < path_idx; i++) {
      base_index += capacities[i];
    }
    for (int sample_idx = blockIdx.x;
         sample_idx < path_loads[path_idx] && sample_idx < capacities[path_idx];
         sample_idx += gridDim.x) {
      int global_idx = sample_idx * path_num + path_idx;
      int out_data_index = (base_index + sample_idx) * sample_dim;
      int in_data_index = route_indices[global_idx] * sample_dim;

      for (int j = threadIdx.x; j < sample_dim; j += 1024) {
        out_data[out_data_index + j] = in_data[in_data_index + j];
      }
    }
  }
}

__global__ void __launch_bounds__(1024)
    route_with_in_indices(float* __restrict__ in_data /*[sample_num x sample_dim]*/,
                          float* __restrict__ out_data /*[?load*path_num x sample_dim]*/,
                          float* __restrict__ gates /*[sample_num x path_num]*/,
                          int* __restrict__ route_indices /*[sample_num x path_num]*/,
                          int* __restrict__ path_loads /*[path_num]*/,
                          int* __restrict__ capacities /*[path_num] */, int sample_dim,
                          int path_num) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] blockIdx.y = 4
  // [thread_extent] threadIdx.x = 1024

  for (int path_idx = blockIdx.y; path_idx < path_num; path_idx += gridDim.y) {
    int base_index = 0;
    for (int i = 0; i < path_idx; i++) {
      base_index += capacities[i];
    }
    for (int sample_idx = blockIdx.x;
         sample_idx < path_loads[path_idx] && sample_idx < capacities[path_idx];
         sample_idx += gridDim.x) {
      int global_idx = sample_idx * path_num + path_idx;
      int out_data_index = (base_index + sample_idx) * sample_dim;
      int in_data_index = route_indices[global_idx] * sample_dim;

      for (int j = threadIdx.x; j < sample_dim; j += 1024) {
        out_data[out_data_index + j] = in_data[in_data_index + j] * gates[global_idx];
      }
    }
  }
}

__global__ void __launch_bounds__(1024) atomic_no_transform_route_back_with_out_indices(
    float* __restrict__ in_data /*[sample_num x sample_dim]*/,
    float* __restrict__ out_data /*[?load*path_num x sample_dim]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ path_loads /*[path_num]*/, int* __restrict__ capacities /*[path_num] */,
    int sample_dim, int path_num) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] blockIdx.y = 4
  // [thread_extent] threadIdx.x = 1024

  for (int path_idx = blockIdx.y; path_idx < path_num; path_idx += gridDim.y) {
    int base_index = 0;
    for (int i = 0; i < path_idx; i++) {
      base_index += capacities[i];
    }
    for (int sample_idx = blockIdx.x;
         sample_idx < path_loads[path_idx] && sample_idx < capacities[path_idx];
         sample_idx += gridDim.x) {
      int global_idx = sample_idx * path_num + path_idx;
      int in_data_index = (base_index + sample_idx) * sample_dim;
      int out_data_index = route_indices[global_idx] * sample_dim;

      for (int j = threadIdx.x; j < sample_dim; j += 1024) {
        atomicAdd(&out_data[out_data_index + j], in_data[in_data_index + j]);
      }
    }
  }
}

__global__ void __launch_bounds__(1024) no_transform_route_back_with_out_indices(
    float* __restrict__ in_data /*[?load*path_num x sample_dim]*/,
    float* __restrict__ out_data /*[sample_num x sample_dim]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ path_loads /*[path_num]*/, int* __restrict__ capacities /*[path_num] */,
    int sample_dim, int path_num) {
  // [thread_extent] blockIdx.x = 512
  // [thread_extent] threadIdx.x = 1024

  int base_index = 0;
  for (int path_idx = 0; path_idx < path_num; path_idx++) {
    for (int sample_idx = blockIdx.x;
         sample_idx < path_loads[path_idx] && sample_idx < capacities[path_idx];
         sample_idx += gridDim.x) {
      int global_idx = sample_idx * path_num + path_idx;
      int in_data_index = (base_index + sample_idx) * sample_dim;
      int out_data_index = route_indices[global_idx] * sample_dim;

      for (int j = threadIdx.x; j < sample_dim; j += 1024) {
        out_data[out_data_index + j] += in_data[in_data_index + j];
      }
    }
    base_index += capacities[path_idx];
  }
}

__global__ void __launch_bounds__(1024) atomic_route_back_with_out_indices(
    float* __restrict__ in_data /*[?load*path_num x sample_dim]*/,
    float* __restrict__ out_data /*[sample_num x sample_dim]*/,
    float* __restrict__ gates /*[sample_num x path_num]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ path_loads /*[path_num]*/, int* __restrict__ capacities /*[path_num] */,
    int sample_dim, int path_num) {
  // [thread_extent] blockIdx.x = 128
  // [thread_extent] blockIdx.y = 4
  // [thread_extent] threadIdx.x = 1024

  for (int path_idx = blockIdx.y; path_idx < path_num; path_idx += gridDim.y) {
    int base_index = 0;
    for (int i = 0; i < path_idx; i++) {
      base_index += capacities[i];
    }
    for (int sample_idx = blockIdx.x;
         sample_idx < path_loads[path_idx] && sample_idx < capacities[path_idx];
         sample_idx += gridDim.x) {
      int global_idx = sample_idx * path_num + path_idx;
      int in_data_index = (base_index + sample_idx) * sample_dim;
      int out_data_index = route_indices[global_idx] * sample_dim;

      for (int j = threadIdx.x; j < sample_dim; j += 1024) {
        atomicAdd(&out_data[out_data_index + j], in_data[in_data_index + j] * gates[global_idx]);
      }
    }
  }
}

__global__ void __launch_bounds__(1024)
    route_back_with_out_indices(float* __restrict__ in_data /*[?load*path_num x sample_dim]*/,
                                float* __restrict__ out_data /*[sample_num x sample_dim]*/,
                                float* __restrict__ gates /*[sample_num x path_num]*/,
                                int* __restrict__ route_indices /*[sample_num x path_num]*/,
                                int* __restrict__ path_loads /*[path_num]*/,
                                int* __restrict__ capacities /*[path_num] */, int sample_dim,
                                int path_num) {
  // [thread_extent] blockIdx.x = 512
  // [thread_extent] threadIdx.x = 1024

  int base_index = 0;
  for (int path_idx = 0; path_idx < path_num; path_idx++) {
    for (int sample_idx = blockIdx.x;
         sample_idx < path_loads[path_idx] && sample_idx < capacities[path_idx];
         sample_idx += gridDim.x) {
      int global_idx = sample_idx * path_num + path_idx;
      int in_data_index = (base_index + sample_idx) * sample_dim;
      int out_data_index = route_indices[global_idx] * sample_dim;

      for (int j = threadIdx.x; j < sample_dim; j += 1024) {
        out_data[out_data_index + j] += in_data[in_data_index + j] * gates[global_idx];
      }
    }
    base_index += capacities[path_idx];
  }
}

void RouteWithInDataIndices(float* in_data /*[sample_num x sample_dim]*/,
                            float* outdata /*[?load*path_num x sample_dim]*/,
                            float* gates /*[sample_num x path_num]*/,
                            int* route_indices /*[sample_num x path_num]*/,
                            int* path_loads /*[path_num]*/, int* capacities /*[path_num]*/,
                            int sample_dim, int path_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  constexpr dim3 grid_size(128, 4);
  if (gates == nullptr) {
    no_transform_route_with_in_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, route_indices, path_loads, capacities, sample_dim, path_num);
  } else {
    route_with_in_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, gates, route_indices, path_loads, capacities, sample_dim, path_num);
  }
}

void RouteBackWithOutDataIndices(float* in_data /*[?load*path_num x sample_dim]*/,
                                 float* outdata /*[sample_num x sample_dim]*/,
                                 float* gates /*[sample_num x path_num]*/,
                                 int* route_indices /*[sample_num x path_num]*/,
                                 int* loads /*[path_num]*/, int sample_num, int sample_dim,
                                 int path_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (gates == nullptr) {
    no_transform_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, route_indices, loads, sample_num, sample_dim, path_num);
  } else {
    combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, gates, route_indices, loads, sample_num, sample_dim, path_num);
  }
}

void DispatchWithDstIndices1D(float* src_data /*[sample_num x sample_dim]*/,
                              float* dst_data /*[?load*path_num x sample_dim]*/,
                              float* gates /*[sample_num x path_num]*/,
                              int* route_indices /*[sample_num x path_num]*/,
                              int* loads /*[path_num]*/, const int& sample_num,
                              const int& sample_dim, const int& path_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  dim3 grid_size(512, path_num);
  if (gates == nullptr) {
    no_transform_dispatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
        src_data, dst_data, route_indices, loads, sample_num, sample_dim, path_num);
  } else {
    dipatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
        src_data, dst_data, gates, route_indices, loads, sample_num, sample_dim, path_num);
  }
}

void DispatchWithDstIndices2D(float* src_data /*[sample_num x sample_dim]*/,
                              float* dst_data /*[?load*path_num x sample_dim]*/,
                              int* route_indices /*[sample_num x path_num]*/,
                              int* loads /*[path_num]*/, const int& sample_num,
                              const int& sample_dim, const int& path_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  dim3 grid_size(512, path_num);
  dispatch_with_dst_indices_2d<<<grid_size, block_size, 0, stream>>>(
      src_data, dst_data, route_indices, loads, sample_num, sample_dim, path_num);
}

void CombineWithSrcIndices(float* src_data /*[?load*path_num x sample_dim]*/,
                           float* dst_data /*[sample_num x sample_dim]*/,
                           float* gates /*[sample_num x path_num]*/,
                           int* route_indices /*[sample_num x path_num]*/,
                           int* loads /*[path_num]*/, const int& sample_num, const int& sample_dim,
                           const int& path_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (gates == nullptr) {
    no_transform_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
        src_data, dst_data, route_indices, loads, sample_num, sample_dim, path_num);
  } else {
    combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
        src_data, dst_data, gates, route_indices, loads, sample_num, sample_dim, path_num);
  }
}

}  // namespace router
}  // namespace brt