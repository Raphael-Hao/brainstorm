#include <brt/router/route.h>
#include <stdio.h>

#include <cuda/std/type_traits>

namespace brt {
namespace router {
template <typename dtype>
__global__ void __launch_bounds__(1024) dispatch_with_dst_indices_2d(
    dtype* __restrict__ in_data /*[path_num x sample_num x sample_size]*/,
    dtype* __restrict__ out_data /*[?load*path_num x sample_size]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int sample_num,
    int sample_size,
    int path_num) {
  in_data += sample_num * sample_size * blockIdx.x;

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
    for (int j = threadIdx.x; j < sample_size; j += 1024) {
      out_data[global_dst * sample_size + j] = in_data[i * sample_size + j];
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) padded_dispatch_with_dst_indices_2d(
    dtype* __restrict__ in_data /*[path_num x sample_num x sample_size]*/,
    dtype* __restrict__ out_data /*[?load*path_num x sample_size]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int capacity,
    int sample_num,
    int sample_size,
    int path_num) {
  in_data += sample_num * sample_size * blockIdx.x;

  int load_start = capacity * blockIdx.y;

  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst = route_indices[route_index];

    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }

    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < sample_size; j += 1024) {
      out_data[global_dst * sample_size + j] = in_data[i * sample_size + j];
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    dispatch_with_dst_indices(dtype* __restrict__ in_data /*[sample_num x sample_size]*/,
                              dtype* __restrict__ out_data /*[?load*path_num x sample_size]*/,
                              int* __restrict__ route_indices /*[sample_num x path_num]*/,
                              int* __restrict__ loads /*[path_num]*/,
                              int sample_num,
                              int sample_size,
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
    for (int j = threadIdx.x; j < sample_size; j += 1024) {
      out_data[global_dst * sample_size + j] = in_data[i * sample_size + j];
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) padded_dispatch_with_dst_indices(
    dtype* __restrict__ in_data /*[sample_num x sample_size]*/,
    dtype* __restrict__ out_data /*[?load*path_num x sample_size]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int capacity,
    int sample_num,
    int sample_size,
    int path_num) {
  int load_start = blockIdx.y * capacity;

  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst = route_indices[route_index];

    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }

    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < sample_size; j += 1024) {
      out_data[global_dst * sample_size + j] = in_data[i * sample_size + j];
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) weighted_dipatch_with_dst_indices(
    dtype* __restrict__ in_data /*[sample_num x sample_size]*/,
    dtype* __restrict__ out_data /*[?load*path_num x sample_size]*/,
    dtype* __restrict__ gates /*[sample_num x path_num]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int sample_num,
    int sample_size,
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
    for (int j = threadIdx.x; j < sample_size; j += 1024) {
      if (::cuda::std::is_same_v<dtype, float>) {
        out_data[global_dst * sample_size + j] = in_data[i * sample_size + j] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) padded_weighted_dipatch_with_dst_indices(
    dtype* __restrict__ in_data /*[sample_num x sample_size]*/,
    dtype* __restrict__ out_data /*[?load*path_num x sample_size]*/,
    dtype* __restrict__ gates /*[sample_num x path_num]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int capacity,
    int sample_num,
    int sample_size,
    int path_num) {
  int load_start = blockIdx.y * capacity;

  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;

    int local_dst = route_indices[route_index];
    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }

    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < sample_size; j += 1024) {
      out_data[global_dst * sample_size + j] = in_data[i * sample_size + j] * gates[route_index];
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    combine_with_src_indices(dtype* __restrict__ in_data /*[?load*path_num x sample_size]*/,
                             dtype* __restrict__ out_data /*[sample_num x sample_size]*/,
                             int* __restrict__ route_indices /*[sample_num x path_num]*/,
                             int* __restrict__ loads /*[path_num]*/,
                             int sample_num,
                             int sample_size,
                             int path_num) {
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
      for (int k = threadIdx.x; k < sample_size; k += 1024) {
        out_data[i * sample_size + k] += in_data[global_dst * sample_size + k];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    padded_combine_with_src_indices(dtype* __restrict__ in_data /*[?load*path_num x sample_size]*/,
                                    dtype* __restrict__ out_data /*[sample_num x sample_size]*/,
                                    int* __restrict__ route_indices /*[sample_num x path_num]*/,
                                    int* __restrict__ loads /*[path_num]*/,
                                    int capacity,
                                    int sample_num,
                                    int sample_size,
                                    int path_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];

      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }

      int global_dst = local_dst - 1 + j * capacity;
      for (int k = threadIdx.x; k < sample_size; k += 1024) {
        out_data[i * sample_size + k] += in_data[global_dst * sample_size + k];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) weighted_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x sample_size]*/,
    dtype* __restrict__ out_data /*[sample_num x sample_size]*/,
    dtype* __restrict__ gates /*[sample_num x path_num]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int sample_num,
    int sample_size,
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
      for (int k = threadIdx.x; k < sample_size; k += 1024) {
        out_data[i * sample_size + k] += in_data[global_dst * sample_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) padded_weighted_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x sample_size]*/,
    dtype* __restrict__ out_data /*[sample_num x sample_size]*/,
    dtype* __restrict__ gates /*[sample_num x path_num]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int capacity,
    int sample_num,
    int sample_size,
    int path_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];
      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }
      int global_dst = local_dst - 1 + j * capacity;
      for (int k = threadIdx.x; k < sample_size; k += 1024) {
        out_data[i * sample_size + k] += in_data[global_dst * sample_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) residual_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x sample_size]*/,
    dtype* __restrict__ out_data /*[sample_num x sample_size]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int sample_num,
    int sample_size,
    int path_num) {
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
      for (int k = threadIdx.x; k < sample_size; k += 1024) {
        out_data[i * sample_size + k] = in_data[global_dst * sample_size + k];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) residual_padded_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x sample_size]*/,
    dtype* __restrict__ out_data /*[sample_num x sample_size]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int capacity,
    int sample_num,
    int sample_size,
    int path_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];

      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }

      int global_dst = local_dst - 1 + j * capacity;
      for (int k = threadIdx.x; k < sample_size; k += 1024) {
        out_data[i * sample_size + k] = in_data[global_dst * sample_size + k];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) residual_weighted_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x sample_size]*/,
    dtype* __restrict__ out_data /*[sample_num x sample_size]*/,
    dtype* __restrict__ gates /*[sample_num x path_num]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int sample_num,
    int sample_size,
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
      for (int k = threadIdx.x; k < sample_size; k += 1024) {
        out_data[i * sample_size + k] = in_data[global_dst * sample_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) residual_padded_weighted_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x sample_size]*/,
    dtype* __restrict__ out_data /*[sample_num x sample_size]*/,
    dtype* __restrict__ gates /*[sample_num x path_num]*/,
    int* __restrict__ route_indices /*[sample_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int capacity,
    int sample_num,
    int sample_size,
    int path_num) {
  for (int i = blockIdx.x; i < sample_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];
      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }
      int global_dst = local_dst - 1 + j * capacity;
      for (int k = threadIdx.x; k < sample_size; k += 1024) {
        out_data[i * sample_size + k] = in_data[global_dst * sample_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
void DispatchWithDstIndices1D(void* src_data /*[sample_num x sample_size]*/,
                              void* dst_data /*[?load*path_num x sample_size]*/,
                              void* gates /*[sample_num x path_num]*/,
                              int* route_indices /*[sample_num x path_num]*/,
                              int* loads /*[path_num]*/,
                              const int& capacity,
                              const int& sample_num,
                              const int& sample_size,
                              const int& path_num,
                              cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512, path_num);
  if (capacity == 0) {
    if (gates == nullptr) {
      dispatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, sample_num, sample_size, path_num);
      // CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      weighted_dipatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, sample_num, sample_size,
          path_num);
      // CUDA_CHECK(cudaDeviceSynchronize());
    }
  } else {
    if (gates == nullptr) {
      padded_dispatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, capacity, sample_num, sample_size,
          path_num);
      // CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      padded_weighted_dipatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, capacity, sample_num,
          sample_size, path_num);
      // CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

template <typename dtype>
void DispatchWithDstIndices2D(void* src_data /*[sample_num x sample_size]*/,
                              void* dst_data /*[?load*path_num x sample_size]*/,
                              int* route_indices /*[sample_num x path_num]*/,
                              int* loads /*[path_num]*/,
                              const int& capacity,
                              const int& sample_num,
                              const int& sample_size,
                              const int& path_num,
                              cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512, path_num);
  if (capacity == 0) {
    dispatch_with_dst_indices_2d<<<grid_size, block_size, 0, stream>>>(
        src_data_ptr, dst_data_ptr, route_indices, loads, sample_num, sample_size, path_num);
  } else {
    padded_dispatch_with_dst_indices_2d<<<grid_size, block_size, 0, stream>>>(
        src_data_ptr, dst_data_ptr, route_indices, loads, capacity, sample_num, sample_size,
        path_num);
  }
}

template <typename dtype>
void CombineWithSrcIndices(void* src_data /*[?load*path_num x sample_size]*/,
                           void* dst_data /*[sample_num x sample_size]*/,
                           void* gates /*[sample_num x path_num]*/,
                           int* route_indices /*[sample_num x path_num]*/,
                           int* loads /*[path_num]*/,
                           const int& capacity,
                           const int& sample_num,
                           const int& sample_size,
                           const int& path_num,
                           cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (capacity == 0) {
    if (gates == nullptr) {
      combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, sample_num, sample_size, path_num);
    } else {
      weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, sample_num, sample_size,
          path_num);
    }
  } else {
    if (gates == nullptr) {
      padded_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, capacity, sample_num, sample_size,
          path_num);
    } else {
      padded_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, capacity, sample_num,
          sample_size, path_num);
    }
  }
}

template <typename dtype>
void ResidualCombineWithSrcIndices(void* src_data /*[?load*path_num x sample_size]*/,
                                   void* dst_data /*[sample_num x sample_size]*/,
                                   void* gates /*[sample_num x path_num]*/,
                                   int* route_indices /*[sample_num x path_num]*/,
                                   int* loads /*[path_num]*/,
                                   const int& capacity,
                                   const int& sample_num,
                                   const int& sample_size,
                                   const int& path_num,
                                   cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (capacity == 0) {
    if (gates == nullptr) {
      residual_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, sample_num, sample_size, path_num);
    } else {
      residual_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, sample_num, sample_size,
          path_num);
    }
  } else {
    if (gates == nullptr) {
      residual_padded_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, capacity, sample_num, sample_size,
          path_num);
    } else {
      residual_padded_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, capacity, sample_num,
          sample_size, path_num);
    }
  }
}

// explicit instantiation

template void DispatchWithDstIndices1D<float>(void* src_data /*[sample_num, sample_size]*/,
                                              void* dst_data /*[total_loads, sample_size]*/,
                                              void* gates /*[sample_num, dst_num]*/,
                                              int* route_indices /*[sample_num, dst_num]*/,
                                              int* loads /*[dst_num]*/,
                                              const int& capacity,
                                              const int& sample_num,
                                              const int& sample_size,
                                              const int& path_num,
                                              cudaStream_t stream);

template void DispatchWithDstIndices2D<float>(void* src_data /*[sample_num, sample_size]*/,
                                              void* dst_data /*[total_loads, sample_size]*/,
                                              int* route_indices /*[sample_num, dst_num]*/,
                                              int* loads /*[dst_num]*/,
                                              const int& capacity,
                                              const int& sample_num,
                                              const int& sample_size,
                                              const int& path_num,
                                              cudaStream_t stream);

template void CombineWithSrcIndices<float>(void* src_data /*[total_loads, sample_size]*/,
                                           void* dst_data /*[sample_num, sample_size]*/,
                                           void* gates /*[sample_num, dst_num]*/,
                                           int* route_indices /*[sample_num, dst_num]*/,
                                           int* loads /*[dst_num]*/,
                                           const int& capacity,
                                           const int& sample_num,
                                           const int& sample_size,
                                           const int& path_num,
                                           cudaStream_t stream);

template void ResidualCombineWithSrcIndices<float>(void* src_data /*[total_loads, sample_size]*/,
                                                   void* dst_data /*[sample_num, sample_size]*/,
                                                   void* gates /*[sample_num x path_num]*/,
                                                   int* route_indices /*[sample_num x path_num]*/,
                                                   int* loads /*[path_num]*/,
                                                   const int& capacity,
                                                   const int& sample_num,
                                                   const int& sample_size,
                                                   const int& path_num,
                                                   cudaStream_t stream);

template void DispatchWithDstIndices1D<__half2>(void* src_data /*[sample_num, sample_size]*/,
                                                void* dst_data /*[total_loads, sample_size]*/,
                                                void* gates /*[sample_num, dst_num]*/,
                                                int* route_indices /*[sample_num, dst_num]*/,
                                                int* loads /*[dst_num]*/,
                                                const int& capacity,
                                                const int& sample_num,
                                                const int& sample_size,
                                                const int& path_num,
                                                cudaStream_t stream);

template void DispatchWithDstIndices2D<__half2>(void* src_data /*[sample_num, sample_size]*/,
                                                void* dst_data /*[total_loads, sample_size]*/,
                                                int* route_indices /*[sample_num, dst_num]*/,
                                                int* loads /*[dst_num]*/,
                                                const int& capacity,
                                                const int& sample_num,
                                                const int& sample_size,
                                                const int& path_num,
                                                cudaStream_t stream);

template void CombineWithSrcIndices<__half2>(void* src_data /*[total_loads, sample_size]*/,
                                             void* dst_data /*[sample_num, sample_size]*/,
                                             void* gates /*[sample_num, dst_num]*/,
                                             int* route_indices /*[sample_num, dst_num]*/,
                                             int* loads /*[dst_num]*/,
                                             const int& capacity,
                                             const int& sample_num,
                                             const int& sample_size,
                                             const int& path_num,
                                             cudaStream_t stream);

template void ResidualCombineWithSrcIndices<__half2>(void* src_data /*[total_loads, sample_size]*/,
                                                     void* dst_data /*[sample_num, sample_size]*/,
                                                     void* gates /*[sample_num, path_num]*/,
                                                     int* route_indices /*[sample_num, path_num]*/,
                                                     int* loads /*[path_num]*/,
                                                     const int& capacity,
                                                     const int& sample_num,
                                                     const int& sample_size,
                                                     const int& path_num,
                                                     cudaStream_t stream);

}  // namespace router
}  // namespace brt