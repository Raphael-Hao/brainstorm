/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <brt/router/route.h>
#include <cuda_fp16.h>
#include <dmlc/common.h>

#include <cuda/std/type_traits>
#include <exception>

namespace brt {
namespace router {
template <typename dtype, bool max_padding, bool is_tag_raouting>
__global__ void __launch_bounds__(1024)
    dispatch_with_dst_indices_2d(dtype* __restrict__ in_data /*[path_num x cell_num x cell_size]*/,
                                 dtype* __restrict__ out_data /*[?load*path_num x cell_size]*/,
                                 int* __restrict__ in_tags,
                                 int* __restrict__ out_tags,
                                 int* __restrict__ route_indices /*[cell_num x path_num]*/,
                                 int* __restrict__ loads /*[path_num]*/,
                                 int cell_num,
                                 int cell_size,
                                 int path_num,
                                 int cell_num_per_path) {
  in_data += cell_num * cell_size * blockIdx.y;
  int load_start = 0;
  if (max_padding) {
    load_start = cell_num_per_path * blockIdx.y;
  } else {
    for (int i = 0; i < blockIdx.y; i++) {
      load_start += loads[i];
    }
  }

  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst = route_indices[route_index];

    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }

    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < cell_size; j += 1024) {
      out_data[global_dst * cell_size + j] = in_data[i * cell_size + j];
    }

    if (is_tag_raouting) {
      if (threadIdx.x == 0) {
        out_tags[global_dst] = in_tags[i];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    dispatch_with_dst_indices(dtype* __restrict__ in_data /*[cell_num x cell_size]*/,
                              dtype* __restrict__ out_data /*[?load*path_num x cell_size]*/,
                              int* __restrict__ route_indices /*[cell_num x path_num]*/,
                              int* __restrict__ loads /*[path_num]*/,
                              int cell_num,
                              int cell_size,
                              int path_num) {
  int load_start = 0;
  for (int i = 0; i < blockIdx.y; i++) {
    load_start += loads[i];
  }

  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst = route_indices[route_index];

    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }

    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < cell_size; j += 1024) {
      out_data[global_dst * cell_size + j] = in_data[i * cell_size + j];
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    padded_dispatch_with_dst_indices(dtype* __restrict__ in_data /*[cell_num x cell_size]*/,
                                     dtype* __restrict__ out_data /*[?load*path_num x cell_size]*/,
                                     int* __restrict__ route_indices /*[cell_num x path_num]*/,
                                     int* __restrict__ loads /*[path_num]*/,
                                     int cell_num_per_path,
                                     int cell_num,
                                     int cell_size,
                                     int path_num) {
  int load_start = blockIdx.y * cell_num_per_path;

  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst = route_indices[route_index];

    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }

    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < cell_size; j += 1024) {
      out_data[global_dst * cell_size + j] = in_data[i * cell_size + j];
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    weighted_dipatch_with_dst_indices(dtype* __restrict__ in_data /*[cell_num x cell_size]*/,
                                      dtype* __restrict__ out_data /*[?load*path_num x cell_size]*/,
                                      dtype* __restrict__ gates /*[cell_num x path_num]*/,
                                      int* __restrict__ route_indices /*[cell_num x path_num]*/,
                                      int* __restrict__ loads /*[path_num]*/,
                                      int cell_num,
                                      int cell_size,
                                      int path_num) {
  int load_start = 0;
  for (int i = 0; i < blockIdx.y; i++) {
    load_start += loads[i];
  }

  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst = route_indices[route_index];

    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }

    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < cell_size; j += 1024) {
      if (::cuda::std::is_same_v<dtype, float>) {
        out_data[global_dst * cell_size + j] = in_data[i * cell_size + j] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) padded_weighted_dipatch_with_dst_indices(
    dtype* __restrict__ in_data /*[cell_num x cell_size]*/,
    dtype* __restrict__ out_data /*[?load*path_num x cell_size]*/,
    dtype* __restrict__ gates /*[cell_num x path_num]*/,
    int* __restrict__ route_indices /*[cell_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int cell_num_per_path,
    int cell_num,
    int cell_size,
    int path_num) {
  int load_start = blockIdx.y * cell_num_per_path;

  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;

    int local_dst = route_indices[route_index];
    if (local_dst == 0 || local_dst > loads[blockIdx.y]) {
      continue;
    }

    int global_dst = local_dst - 1 + load_start;
    for (int j = threadIdx.x; j < cell_size; j += 1024) {
      out_data[global_dst * cell_size + j] = in_data[i * cell_size + j] * gates[route_index];
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    combine_with_src_indices(dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
                             dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
                             int* __restrict__ route_indices /*[cell_num x path_num]*/,
                             int* __restrict__ loads /*[path_num]*/,
                             int cell_num,
                             int cell_size,
                             int path_num) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
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
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] += in_data[global_dst * cell_size + k];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    padded_combine_with_src_indices(dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
                                    dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
                                    int* __restrict__ route_indices /*[cell_num x path_num]*/,
                                    int* __restrict__ loads /*[path_num]*/,
                                    int cell_num_per_path,
                                    int cell_num,
                                    int cell_size,
                                    int path_num) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];

      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }

      int global_dst = local_dst - 1 + j * cell_num_per_path;
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] += in_data[global_dst * cell_size + k];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    weighted_combine_with_src_indices(dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
                                      dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
                                      dtype* __restrict__ gates /*[cell_num x path_num]*/,
                                      int* __restrict__ route_indices /*[cell_num x path_num]*/,
                                      int* __restrict__ loads /*[path_num]*/,
                                      int cell_num,
                                      int cell_size,
                                      int path_num) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
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
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] += in_data[global_dst * cell_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) padded_weighted_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
    dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
    dtype* __restrict__ gates /*[cell_num x path_num]*/,
    int* __restrict__ route_indices /*[cell_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int cell_num_per_path,
    int cell_num,
    int cell_size,
    int path_num) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];
      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }
      int global_dst = local_dst - 1 + j * cell_num_per_path;
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] += in_data[global_dst * cell_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024)
    residual_combine_with_src_indices(dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
                                      dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
                                      int* __restrict__ route_indices /*[cell_num x path_num]*/,
                                      int* __restrict__ loads /*[path_num]*/,
                                      int cell_num,
                                      int cell_size,
                                      int path_num) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
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
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] = in_data[global_dst * cell_size + k];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) residual_padded_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
    dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
    int* __restrict__ route_indices /*[cell_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int cell_num_per_path,
    int cell_num,
    int cell_size,
    int path_num) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];

      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }

      int global_dst = local_dst - 1 + j * cell_num_per_path;
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] = in_data[global_dst * cell_size + k];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) residual_weighted_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
    dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
    dtype* __restrict__ gates /*[cell_num x path_num]*/,
    int* __restrict__ route_indices /*[cell_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int cell_num,
    int cell_size,
    int path_num) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
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
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] = in_data[global_dst * cell_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
__global__ void __launch_bounds__(1024) residual_padded_weighted_combine_with_src_indices(
    dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
    dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
    dtype* __restrict__ gates /*[cell_num x path_num]*/,
    int* __restrict__ route_indices /*[cell_num x path_num]*/,
    int* __restrict__ loads /*[path_num]*/,
    int cell_num_per_path,
    int cell_num,
    int cell_size,
    int path_num) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_dst = route_indices[route_index];
      if (local_dst == 0 || local_dst > loads[j]) {
        continue;
      }
      int global_dst = local_dst - 1 + j * cell_num_per_path;
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] = in_data[global_dst * cell_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
void DispatchWithDstIndices1D(void* src_data /*[cell_num x cell_size]*/,
                              void* dst_data /*[?load*path_num x cell_size]*/,
                              void* gates /*[cell_num x path_num]*/,
                              int* route_indices /*[cell_num x path_num]*/,
                              int* loads /*[path_num]*/,
                              const int& cell_num_per_path,
                              const int& cell_num,
                              const int& cell_size,
                              const int& path_num,
                              cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512, path_num);
  if (cell_num_per_path == 0) {
    if (gates == nullptr) {
      dispatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, cell_num, cell_size, path_num);
      // CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      weighted_dipatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
          path_num);
      // CUDA_CHECK(cudaDeviceSynchronize());
    }
  } else {
    if (gates == nullptr) {
      padded_dispatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, cell_num_per_path, cell_num, cell_size,
          path_num);
      // CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      padded_weighted_dipatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num_per_path, cell_num,
          cell_size, path_num);
      // CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

template <typename dtype>
void DispatchWithDstIndices2D(void* src_data /*[cell_num x cell_size]*/,
                              void* dst_data /*[?load*path_num x cell_size]*/,
                              int* route_indices /*[cell_num x path_num]*/,
                              int* loads /*[path_num]*/,
                              const int& cell_num_per_path,
                              const int& cell_num,
                              const int& cell_size,
                              const int& path_num,
                              cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512, path_num);
  if (cell_num_per_path == 0) {
    dispatch_with_dst_indices_2d<dtype, false, false><<<grid_size, block_size, 0, stream>>>(
        src_data_ptr, dst_data_ptr, nullptr, nullptr, route_indices, loads, cell_num, cell_size,
        path_num, cell_num_per_path);
  } else {
    dispatch_with_dst_indices_2d<dtype, true, false><<<grid_size, block_size, 0, stream>>>(
        src_data_ptr, dst_data_ptr, nullptr, nullptr, route_indices, loads, cell_num, cell_size,
        path_num, cell_num_per_path);
  }
}

template <typename dtype>
void CombineWithSrcIndices(void* src_data /*[?load*path_num x cell_size]*/,
                           void* dst_data /*[cell_num x cell_size]*/,
                           void* gates /*[cell_num x path_num]*/,
                           int* route_indices /*[cell_num x path_num]*/,
                           int* loads /*[path_num]*/,
                           const int& cell_num_per_path,
                           const int& cell_num,
                           const int& cell_size,
                           const int& path_num,
                           cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (cell_num_per_path == 0) {
    if (gates == nullptr) {
      combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, cell_num, cell_size, path_num);
    } else {
      weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
          path_num);
    }
  } else {
    if (gates == nullptr) {
      padded_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, cell_num_per_path, cell_num, cell_size,
          path_num);
    } else {
      padded_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num_per_path, cell_num,
          cell_size, path_num);
    }
  }
}

template <typename dtype>
void ResidualCombineWithSrcIndices(void* src_data /*[?load*path_num x cell_size]*/,
                                   void* dst_data /*[cell_num x cell_size]*/,
                                   void* gates /*[cell_num x path_num]*/,
                                   int* route_indices /*[cell_num x path_num]*/,
                                   int* loads /*[path_num]*/,
                                   const int& cell_num_per_path,
                                   const int& cell_num,
                                   const int& cell_size,
                                   const int& path_num,
                                   cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (cell_num_per_path == 0) {
    if (gates == nullptr) {
      residual_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, cell_num, cell_size, path_num);
    } else {
      residual_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
          path_num);
    }
  } else {
    if (gates == nullptr) {
      residual_padded_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, route_indices, loads, cell_num_per_path, cell_num, cell_size,
          path_num);
    } else {
      residual_padded_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num_per_path, cell_num,
          cell_size, path_num);
    }
  }
}

template <typename dtype>
void DispatchWithIndicesAndLoads(void* src_data /*[cell_num, cell_size]*/,
                                 void* dst_data /*[total_loads, cell_size]*/,
                                 void* gates /*[cell_num, dst_num]*/,
                                 int* route_indices /*[cell_num, dst_num]*/,
                                 int* loads /*[dst_num]*/,
                                 const int& cell_num,
                                 const int& cell_size,
                                 const int& path_num,
                                 cudaStream_t stream,
                                 const int& cell_num_per_path,
                                 bool is_1d_routing,
                                 bool is_dst_index) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);

  constexpr dim3 block_size(1024);
  dim3 grid_size(512, path_num);

  if (is_dst_index) {
    if (is_1d_routing) {
      if (cell_num_per_path == 0) {
        if (gates == nullptr) {
          dispatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, cell_num, cell_size, path_num);
          // CUDA_CHECK(cudaDeviceSynchronize());
        } else {
          weighted_dipatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
              path_num);
          // CUDA_CHECK(cudaDeviceSynchronize());
        }
      } else {
        if (gates == nullptr) {
          padded_dispatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, cell_num_per_path, cell_num,
              cell_size, path_num);
          // CUDA_CHECK(cudaDeviceSynchronize());
        } else {
          padded_weighted_dipatch_with_dst_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num_per_path,
              cell_num, cell_size, path_num);
          // CUDA_CHECK(cudaDeviceSynchronize());
        }
      }
    } else {
      CHECK(gates_ptr == nullptr);
      if (cell_num_per_path == 0) {
        dispatch_with_dst_indices_2d<dtype, false, false><<<grid_size, block_size, 0, stream>>>(
            src_data_ptr, dst_data_ptr, nullptr, nullptr, route_indices, loads, cell_num, cell_size,
            path_num, cell_num_per_path);
      } else {
        dispatch_with_dst_indices_2d<dtype, true, false><<<grid_size, block_size, 0, stream>>>(
            src_data_ptr, dst_data_ptr, nullptr, nullptr, route_indices, loads, cell_num, cell_size,
            path_num, cell_num_per_path);
      }
    }
  } else {
    LOG(FATAL) << "Dispatch with src indices is not supported yet.";
  }
}

template <typename dtype>
void CombineWithIndicesAndLoads(void* src_data /*[total_loads, cell_size]*/,
                                void* dst_data /*[cell_num, cell_size]*/,
                                void* gates /*[cell_num, dst_num]*/,
                                int* route_indices /*[cell_num, dst_num]*/,
                                int* loads /*[dst_num]*/,
                                const int& cell_num,
                                const int& cell_size,
                                const int& path_num,
                                cudaStream_t stream,
                                const int& cell_num_per_path,
                                bool is_residual,
                                bool is_dst_index) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);

  constexpr dim3 block_size(1024);
  dim3 grid_size(512);

  if (is_dst_index) {
    if (is_residual) {
      if (cell_num_per_path == 0) {
        if (gates == nullptr) {
          residual_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, cell_num, cell_size, path_num);
        } else {
          residual_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
              path_num);
        }
      } else {
        if (gates == nullptr) {
          residual_padded_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, cell_num_per_path, cell_num,
              cell_size, path_num);
        } else {
          residual_padded_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num_per_path,
              cell_num, cell_size, path_num);
        }
      }
    } else {
      if (cell_num_per_path == 0) {
        if (gates == nullptr) {
          combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, cell_num, cell_size, path_num);
        } else {
          weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
              path_num);
        }
      } else {
        if (gates == nullptr) {
          padded_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, cell_num_per_path, cell_num,
              cell_size, path_num);
        } else {
          padded_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num_per_path,
              cell_num, cell_size, path_num);
        }
      }
    }
  } else {
    LOG(FATAL) << "Combine with src indices is not supported yet.";
  }
}

// explicit instantiation

template void DispatchWithDstIndices1D<float>(void* src_data /*[cell_num, cell_size]*/,
                                              void* dst_data /*[total_loads, cell_size]*/,
                                              void* gates /*[cell_num, dst_num]*/,
                                              int* route_indices /*[cell_num, dst_num]*/,
                                              int* loads /*[dst_num]*/,
                                              const int& cell_num_per_path,
                                              const int& cell_num,
                                              const int& cell_size,
                                              const int& path_num,
                                              cudaStream_t stream);

template void DispatchWithDstIndices2D<float>(void* src_data /*[cell_num, cell_size]*/,
                                              void* dst_data /*[total_loads, cell_size]*/,
                                              int* route_indices /*[cell_num, dst_num]*/,
                                              int* loads /*[dst_num]*/,
                                              const int& cell_num_per_path,
                                              const int& cell_num,
                                              const int& cell_size,
                                              const int& path_num,
                                              cudaStream_t stream);

template void CombineWithSrcIndices<float>(void* src_data /*[total_loads, cell_size]*/,
                                           void* dst_data /*[cell_num, cell_size]*/,
                                           void* gates /*[cell_num, dst_num]*/,
                                           int* route_indices /*[cell_num, dst_num]*/,
                                           int* loads /*[dst_num]*/,
                                           const int& cell_num_per_path,
                                           const int& cell_num,
                                           const int& cell_size,
                                           const int& path_num,
                                           cudaStream_t stream);

template void ResidualCombineWithSrcIndices<float>(void* src_data /*[total_loads, cell_size]*/,
                                                   void* dst_data /*[cell_num, cell_size]*/,
                                                   void* gates /*[cell_num x path_num]*/,
                                                   int* route_indices /*[cell_num x path_num]*/,
                                                   int* loads /*[path_num]*/,
                                                   const int& cell_num_per_path,
                                                   const int& cell_num,
                                                   const int& cell_size,
                                                   const int& path_num,
                                                   cudaStream_t stream);

template void DispatchWithDstIndices1D<__half2>(void* src_data /*[cell_num, cell_size]*/,
                                                void* dst_data /*[total_loads, cell_size]*/,
                                                void* gates /*[cell_num, dst_num]*/,
                                                int* route_indices /*[cell_num, dst_num]*/,
                                                int* loads /*[dst_num]*/,
                                                const int& cell_num_per_path,
                                                const int& cell_num,
                                                const int& cell_size,
                                                const int& path_num,
                                                cudaStream_t stream);

template void DispatchWithDstIndices2D<__half2>(void* src_data /*[cell_num, cell_size]*/,
                                                void* dst_data /*[total_loads, cell_size]*/,
                                                int* route_indices /*[cell_num, dst_num]*/,
                                                int* loads /*[dst_num]*/,
                                                const int& cell_num_per_path,
                                                const int& cell_num,
                                                const int& cell_size,
                                                const int& path_num,
                                                cudaStream_t stream);

template void CombineWithSrcIndices<__half2>(void* src_data /*[total_loads, cell_size]*/,
                                             void* dst_data /*[cell_num, cell_size]*/,
                                             void* gates /*[cell_num, dst_num]*/,
                                             int* route_indices /*[cell_num, dst_num]*/,
                                             int* loads /*[dst_num]*/,
                                             const int& cell_num_per_path,
                                             const int& cell_num,
                                             const int& cell_size,
                                             const int& path_num,
                                             cudaStream_t stream);

template void ResidualCombineWithSrcIndices<__half2>(void* src_data /*[total_loads, cell_size]*/,
                                                     void* dst_data /*[cell_num, cell_size]*/,
                                                     void* gates /*[cell_num, path_num]*/,
                                                     int* route_indices /*[cell_num, path_num]*/,
                                                     int* loads /*[path_num]*/,
                                                     const int& cell_num_per_path,
                                                     const int& cell_num,
                                                     const int& cell_size,
                                                     const int& path_num,
                                                     cudaStream_t stream);

template void DispatchWithIndicesAndLoads<float>(void* src_data /*[cell_num, cell_size]*/,
                                                 void* dst_data /*[total_loads, cell_size]*/,
                                                 void* gates /*[cell_num, dst_num]*/,
                                                 int* route_indices /*[cell_num, dst_num]*/,
                                                 int* loads /*[dst_num]*/,
                                                 const int& cell_num,
                                                 const int& cell_size,
                                                 const int& path_num,
                                                 cudaStream_t stream,
                                                 const int& cell_num_per_path,
                                                 bool is_1d_routing,
                                                 bool is_dst_index);

template void DispatchWithIndicesAndLoads<__half2>(void* src_data /*[cell_num, cell_size]*/,
                                                   void* dst_data /*[total_loads, cell_size]*/,
                                                   void* gates /*[cell_num, dst_num]*/,
                                                   int* route_indices /*[cell_num, dst_num]*/,
                                                   int* loads /*[dst_num]*/,
                                                   const int& cell_num,
                                                   const int& cell_size,
                                                   const int& path_num,
                                                   cudaStream_t stream,
                                                   const int& cell_num_per_path,
                                                   bool is_1d_routing,
                                                   bool is_dst_index);
template void CombineWithIndicesAndLoads<float>(void* src_data /*[total_loads, cell_size]*/,
                                                void* dst_data /*[cell_num, cell_size]*/,
                                                void* gates /*[cell_num, dst_num]*/,
                                                int* route_indices /*[cell_num, dst_num]*/,
                                                int* loads /*[dst_num]*/,
                                                const int& cell_num,
                                                const int& cell_size,
                                                const int& path_num,
                                                cudaStream_t stream,
                                                const int& cell_num_per_path,
                                                bool is_residual,
                                                bool is_dst_index);

template void CombineWithIndicesAndLoads<__half>(void* src_data /*[total_loads, cell_size]*/,
                                                 void* dst_data /*[cell_num, cell_size]*/,
                                                 void* gates /*[cell_num, dst_num]*/,
                                                 int* route_indices /*[cell_num, dst_num]*/,
                                                 int* loads /*[dst_num]*/,
                                                 const int& cell_num,
                                                 const int& cell_size,
                                                 const int& path_num,
                                                 cudaStream_t stream,
                                                 const int& cell_num_per_path,
                                                 bool is_residual,
                                                 bool is_dst_index);
}  // namespace router
}  // namespace brt