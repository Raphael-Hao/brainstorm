
#include <brt/router/route.h>

namespace brt {
namespace router {

__global__ void __launch_bounds__(1024) no_transform_route_with_out_indices(
    float* __restrict__ in_data /*[sample_num x sample_dim]*/,
    float* __restrict__ out_data /*[?load*dst_num x sample_dim]*/,
    int* __restrict__ route_indices /*[sample_num x dst_num]*/,
    int* __restrict__ dst_loads /*[dst_num]*/, int sample_num, int sample_dim, int dst_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    int route_index = i * dst_num + blockIdx.y;
    int local_dst = route_indices[route_index];
    if (local_dst == 0) {
      continue;
    }
    int global_dst = local_dst - 1;
    for (int j = 0; j < blockIdx.y; j++) {
      global_dst += dst_loads[j];
    }
    for (int j = threadIdx.x; j < sample_dim; j += 1024) {
      out_data[global_dst * sample_dim + j] = in_data[i * sample_dim + j];
    }
  }
}

__global__ void __launch_bounds__(1024)
    route_with_out_indices(float* __restrict__ in_data /*[sample_num x sample_dim]*/,
                           float* __restrict__ out_data /*[?load*dst_num x sample_dim]*/,
                           float* __restrict__ gates /*[sample_num x dst_num]*/,
                           int* __restrict__ route_indices /*[sample_num x dst_num]*/,
                           int* __restrict__ dst_loads /*[dst_num]*/, int sample_num,
                           int sample_dim, int dst_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    int route_index = i * dst_num + blockIdx.y;

    int local_dst = route_indices[route_index];
    if (local_dst == 0) {
      continue;
    }
    int global_dst = local_dst - 1;
    for (int j = 0; j < blockIdx.y; j++) {
      global_dst += dst_loads[j];
    }

    for (int j = threadIdx.x; j < sample_dim; j += 1024) {
      out_data[global_dst * sample_dim + j] = in_data[i * sample_dim + j] * gates[route_index];
    }
  }
}

__global__ void __launch_bounds__(1024) no_transform_route_back_with_in_indices(
    float* __restrict__ in_data /*[?load*dst_num x sample_dim]*/,
    float* __restrict__ out_data /*[sample_num x sample_dim]*/,
    int* __restrict__ route_indices /*[sample_num x dst_num]*/,
    int* __restrict__ dst_loads /*[dst_num]*/, int sample_num, int sample_dim, int dst_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    for (int j = 0; j < dst_num; j++) {
      int route_index = i * dst_num + j;
      int local_dst = route_indices[route_index];
      if (local_dst == 0) {
        continue;
      }
      int global_dst = local_dst - 1;
      for (int k = 0; k < j; k++) {
        global_dst += dst_loads[k];
      }
      for (int k = threadIdx.x; k < sample_dim; k += 1024) {
        out_data[i * sample_dim + k] += in_data[global_dst * sample_dim + k];
      }
    }
  }
}

__global__ void __launch_bounds__(1024)
    route_back_with_in_indices(float* __restrict__ in_data /*[?load*dst_num x sample_dim]*/,
                               float* __restrict__ out_data /*[sample_num x sample_dim]*/,
                               float* __restrict__ gates /*[sample_num x dst_num]*/,
                               int* __restrict__ route_indices /*[sample_num x dst_num]*/,
                               int* __restrict__ dst_loads /*[dst_num]*/, int sample_num,
                               int sample_dim, int dst_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    for (int j = 0; j < dst_num; j++) {
      int route_index = i * dst_num + j;
      int local_dst = route_indices[route_index];
      if (local_dst == 0) {
        continue;
      }
      int global_dst = local_dst - 1;
      for (int k = 0; k < j; k++) {
        global_dst += dst_loads[k];
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
                          float* __restrict__ gates /*[sample_num x dst_num]*/,
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
    float* __restrict__ gates /*[sample_num x dst_num]*/,
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
                                float* __restrict__ gates /*[sample_num x dst_num]*/,
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
                            float* outdata /*[?load*dst_num x sample_dim]*/,
                            float* gates /*[sample_num x dst_num]*/,
                            int* route_indices /*[sample_num x dst_num]*/,
                            int* path_loads /*[dst_num]*/, int* capacities /*[dst_num]*/,
                            int sample_dim, int dst_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  constexpr dim3 grid_size(128, 4);
  if (gates == nullptr) {
    no_transform_route_with_in_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, route_indices, path_loads, capacities, sample_dim, dst_num);
  } else {
    route_with_in_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, gates, route_indices, path_loads, capacities, sample_dim, dst_num);
  }
}

void RouteBackWithOutDataIndices(float* in_data /*[?load*dst_num x sample_dim]*/,
                                 float* outdata /*[sample_num x sample_dim]*/,
                                 float* gates /*[sample_num x dst_num]*/,
                                 int* route_indices /*[sample_num x dst_num]*/,
                                 int* dst_loads /*[dst_num]*/, int sample_num, int sample_dim,
                                 int dst_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (gates == nullptr) {
    no_transform_route_back_with_in_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, route_indices, dst_loads, sample_num, sample_dim, dst_num);
  } else {
    route_back_with_in_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, gates, route_indices, dst_loads, sample_num, sample_dim, dst_num);
  }
}

void RouteWithLocalIndices(float* in_data /*[sample_num x sample_dim]*/,
                           float* outdata /*[?load*dst_num x sample_dim]*/,
                           float* gates /*[sample_num x dst_num]*/,
                           int* route_indices /*[sample_num x dst_num]*/,
                           int* dst_loads /*[dst_num]*/, int sample_num, int sample_dim,
                           int dst_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  dim3 grid_size(512, dst_num);
  if (gates == nullptr) {
    no_transform_route_with_out_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, route_indices, dst_loads, sample_num, sample_dim, dst_num);
  } else {
    route_with_out_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, gates, route_indices, dst_loads, sample_num, sample_dim, dst_num);
  }
}

void RouteBackWithLocalIndices(float* in_data /*[?load*dst_num x sample_dim]*/,
                               float* outdata /*[sample_num x sample_dim]*/,
                               float* gates /*[sample_num x dst_num]*/,
                               int* route_indices /*[sample_num x dst_num]*/,
                               int* dst_loads /*[dst_num]*/, int sample_num, int sample_dim,
                               int dst_num, cudaStream_t stream) {
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (gates == nullptr) {
    no_transform_route_back_with_in_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, route_indices, dst_loads, sample_num, sample_dim, dst_num);
  } else {
    route_back_with_in_indices<<<grid_size, block_size, 0, stream>>>(
        in_data, outdata, gates, route_indices, dst_loads, sample_num, sample_dim, dst_num);
  }
}

}  // namespace router
}  // namespace brt