/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <brt/router/route.h>
#include <cuda_fp16.h>
#include <dmlc/common.h>

#include <exception>

namespace brt {
namespace router {
template <typename dtype, bool weighted, bool max_path_padding, bool tag_generating>
__global__ void __launch_bounds__(1024)
    dispatch_with_seat_indices_1d(dtype* __restrict__ in_data /*[cell_num x cell_size]*/,
                                  dtype* __restrict__ out_data /*[total_load x cell_size]*/,
                                  dtype* __restrict__ gates /*[cell_num x path_num]*/,
                                  int* __restrict__ route_indices /*[cell_num x path_num]*/,
                                  int* __restrict__ loads /*[path_num]*/,
                                  int* __restrict__ old_tags,
                                  int* __restrict__ new_tags,
                                  int cell_num,
                                  int cell_size,
                                  int path_num,
                                  int max_path_load) {
  int load_start = 0;
  if (max_path_padding) {
    load_start = blockIdx.y * max_path_load;
  } else {
    for (int i = 0; i < blockIdx.y; i++) {
      load_start += loads[i];
    }
  }
  int path_load = loads[blockIdx.y];
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_seat_index = route_indices[route_index];

    if (local_seat_index == 0 || local_seat_index > path_load) {
      continue;
    }

    int global_seat_index = local_seat_index - 1 + load_start;
    for (int j = threadIdx.x; j < cell_size; j += 1024) {
      if (weighted) {
        out_data[global_seat_index * cell_size + j] =
            in_data[i * cell_size + j] * gates[route_index];
      } else {
        out_data[global_seat_index * cell_size + j] = in_data[i * cell_size + j];
      }
    }
    if (threadIdx.x == 0 && tag_generating) {
      new_tags[global_seat_index] = old_tags[i];
    }
  }
  if (threadIdx.x == 0 && max_path_padding) {
    loads[blockIdx.y] = max_path_load;
  }
}

template <typename dtype, bool max_path_padding, bool tag_generating>
__global__ void __launch_bounds__(1024)
    dispatch_with_seat_indices_2d(dtype* __restrict__ in_data /*[path_num x cell_num x cell_size]*/,
                                  dtype* __restrict__ out_data /*[?load*path_num x cell_size]*/,
                                  int* __restrict__ route_indices /*[cell_num x path_num]*/,
                                  int* __restrict__ loads /*[path_num]*/,
                                  int* __restrict__ old_tags,
                                  int* __restrict__ new_tags,
                                  int cell_num,
                                  int cell_size,
                                  int path_num,
                                  int max_path_load) {
  in_data += cell_num * cell_size * blockIdx.y;
  int load_start = 0;
  if (max_path_padding) {
    load_start = max_path_load * blockIdx.y;
  } else {
    for (int i = 0; i < blockIdx.y; i++) {
      load_start += loads[i];
    }
  }
  int path_load = loads[blockIdx.y];
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    int route_index = i * path_num + blockIdx.y;
    int local_dst_index = route_indices[route_index];

    if (local_dst_index == 0 || local_dst_index > path_load) {
      continue;
    }

    int global_dst_index = local_dst_index - 1 + load_start;
    for (int j = threadIdx.x; j < cell_size; j += 1024) {
      out_data[global_dst_index * cell_size + j] = in_data[i * cell_size + j];
    }

    if (threadIdx.x == 0 && tag_generating) {
      new_tags[global_dst_index] = old_tags[i];
    }
  }
  if (threadIdx.x == 0 && max_path_padding) {
    loads[blockIdx.y] = max_path_load;
  }
}

template <typename dtype, bool weighted, bool max_path_padding>
__global__ void __launch_bounds__(1024)
    combine_with_seat_indices(dtype* __restrict__ in_data /*[?load*path_num x cell_size]*/,
                              dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
                              dtype* __restrict__ gates /*[cell_num x path_num]*/,
                              int* __restrict__ route_indices /*[cell_num x path_num]*/,
                              int* __restrict__ loads /*[path_num]*/,
                              int cell_num,
                              int cell_size,
                              int path_num,
                              int max_path_load) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_seat = route_indices[route_index];

      if (local_seat == 0 || local_seat > loads[j]) {
        continue;
      }

      int global_seat = local_seat - 1;
      if (max_path_padding) {
        global_seat += max_path_load * j;
      } else {
        for (int k = 0; k < j; k++) {
          global_seat += loads[k];
        }
      }
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        dtype* write_addr = out_data + i * cell_size + k;
        dtype write_back;
        if (weighted) {
          write_back = in_data[global_seat * cell_size + k] * gates[route_index];
        } else {
          write_back = in_data[global_seat * cell_size + k];
        }
        atomicAdd(write_addr, write_back);
      }
    }
  }
}

template <typename dtype, bool max_path_padding>
__global__ void __launch_bounds__(1024)
    init_with_seat_indices(dtype init_scalar /*[?load*path_num x cell_size]*/,
                           dtype* __restrict__ out_data /*[cell_num x cell_size]*/,
                           int* __restrict__ route_indices /*[cell_num x path_num]*/,
                           int* __restrict__ loads /*[path_num]*/,
                           int cell_num,
                           int cell_size,
                           int path_num,
                           int max_path_load) {
  for (int i = blockIdx.x; i < cell_num; i += gridDim.x) {
    for (int j = 0; j < path_num; j++) {
      int route_index = i * path_num + j;
      int local_seat = route_indices[route_index];

      if (local_seat == 0 || local_seat > loads[j]) {
        continue;
      }

      int global_seat = local_seat - 1;
      if (max_path_padding) {
        global_seat += max_path_load * j;
      } else {
        for (int k = 0; k < j; k++) {
          global_seat += loads[k];
        }
      }
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] = init_scalar;
      }
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
                                    int max_path_load,
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

      int global_dst = local_dst - 1 + j * max_path_load;
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
    int max_path_load,
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
      int global_dst = local_dst - 1 + j * max_path_load;
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
    int max_path_load,
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

      int global_dst = local_dst - 1 + j * max_path_load;
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
    int max_path_load,
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
      int global_dst = local_dst - 1 + j * max_path_load;
      for (int k = threadIdx.x; k < cell_size; k += 1024) {
        out_data[i * cell_size + k] = in_data[global_dst * cell_size + k] * gates[route_index];
      }
    }
  }
}

template <typename dtype>
void CombineWithSrcIndices(void* src_data /*[?load*path_num x cell_size]*/,
                           void* dst_data /*[cell_num x cell_size]*/,
                           void* gates /*[cell_num x path_num]*/,
                           int* route_indices /*[cell_num x path_num]*/,
                           int* loads /*[path_num]*/,
                           const int& max_path_load,
                           const int& cell_num,
                           const int& cell_size,
                           const int& path_num,
                           cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (max_path_load == 0) {
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
          src_data_ptr, dst_data_ptr, route_indices, loads, max_path_load, cell_num, cell_size,
          path_num);
    } else {
      padded_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, max_path_load, cell_num,
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
                                   const int& max_path_load,
                                   const int& cell_num,
                                   const int& cell_size,
                                   const int& path_num,
                                   cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);
  constexpr dim3 block_size(1024);
  dim3 grid_size(512);
  if (max_path_load == 0) {
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
          src_data_ptr, dst_data_ptr, route_indices, loads, max_path_load, cell_num, cell_size,
          path_num);
    } else {
      residual_padded_weighted_combine_with_src_indices<<<grid_size, block_size, 0, stream>>>(
          src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, max_path_load, cell_num,
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
                                 int* old_tags,
                                 int* new_tags,
                                 const int& cell_num,
                                 const int& cell_size,
                                 const int& path_num,
                                 const int& max_path_load,
                                 bool is_1d_routing,
                                 bool is_tag_index,
                                 cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);

  constexpr dim3 block_size(1024);
  dim3 grid_size(512, path_num);

  if (!is_tag_index) {
    if (is_1d_routing) {
      if (gates != nullptr) {
        if (max_path_load != 0) {
          if (old_tags != nullptr) {
            dispatch_with_seat_indices_1d<dtype, true, true, true>
                <<<grid_size, block_size, 0, stream>>>(
                    src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, old_tags, new_tags,
                    cell_num, cell_size, path_num, max_path_load);
          } else {
            dispatch_with_seat_indices_1d<dtype, true, true, false>
                <<<grid_size, block_size, 0, stream>>>(
                    src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, old_tags, new_tags,
                    cell_num, cell_size, path_num, max_path_load);
          }
        } else {
          if (old_tags != nullptr) {
            dispatch_with_seat_indices_1d<dtype, true, false, true>
                <<<grid_size, block_size, 0, stream>>>(
                    src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, old_tags, new_tags,
                    cell_num, cell_size, path_num, max_path_load);
          } else {
            dispatch_with_seat_indices_1d<dtype, true, false, false>
                <<<grid_size, block_size, 0, stream>>>(
                    src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, old_tags, new_tags,
                    cell_num, cell_size, path_num, max_path_load);
          }
        }
      } else {
        if (max_path_load != 0) {
          if (old_tags != nullptr) {
            dispatch_with_seat_indices_1d<dtype, false, true, true>
                <<<grid_size, block_size, 0, stream>>>(
                    src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, old_tags, new_tags,
                    cell_num, cell_size, path_num, max_path_load);
          } else {
            dispatch_with_seat_indices_1d<dtype, false, true, false>
                <<<grid_size, block_size, 0, stream>>>(
                    src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, old_tags, new_tags,
                    cell_num, cell_size, path_num, max_path_load);
          }
        } else {
          if (old_tags != nullptr) {
            dispatch_with_seat_indices_1d<dtype, false, false, true>
                <<<grid_size, block_size, 0, stream>>>(
                    src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, old_tags, new_tags,
                    cell_num, cell_size, path_num, max_path_load);
          } else {
            dispatch_with_seat_indices_1d<dtype, false, false, false>
                <<<grid_size, block_size, 0, stream>>>(
                    src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, old_tags, new_tags,
                    cell_num, cell_size, path_num, max_path_load);
          }
        }
      }
    } else {
      CHECK(gates_ptr == nullptr);
      if (max_path_load != 0) {
        if (old_tags != nullptr) {
          dispatch_with_seat_indices_2d<dtype, true, true><<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, old_tags, new_tags, cell_num,
              cell_size, path_num, max_path_load);
        } else {
          dispatch_with_seat_indices_2d<dtype, true, false><<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, old_tags, new_tags, cell_num,
              cell_size, path_num, max_path_load);
        }
      } else {
        if (old_tags != nullptr) {
          dispatch_with_seat_indices_2d<dtype, false, true><<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, old_tags, new_tags, cell_num,
              cell_size, path_num, max_path_load);
        } else {
          dispatch_with_seat_indices_2d<dtype, false, false><<<grid_size, block_size, 0, stream>>>(
              src_data_ptr, dst_data_ptr, route_indices, loads, old_tags, new_tags, cell_num,
              cell_size, path_num, max_path_load);
        }
      }
    }
  } else {
    LOG(FATAL) << "Dispatch with tag indices is not supported yet.";
  }
}

template <typename dtype>
void CombineWithIndicesAndLoads(void* src_data /*[total_loads, cell_size]*/,
                                void* dst_data /*[cell_num, cell_size]*/,
                                void* gates /*[cell_num, dst_num]*/,
                                int* route_indices /*[cell_num, dst_num]*/,
                                int* loads /*[dst_num]*/,
                                int* old_tags,
                                int* new_tags,
                                const int& cell_num,
                                const int& cell_size,
                                const int& path_num,
                                const int& max_path_load,
                                bool is_residual,
                                bool is_tag_index,
                                cudaStream_t stream) {
  dtype* src_data_ptr = static_cast<dtype*>(src_data);
  dtype* dst_data_ptr = static_cast<dtype*>(dst_data);
  dtype* gates_ptr = static_cast<dtype*>(gates);

  constexpr dim3 block_size(1024);
  dim3 grid_size(512);

  if (!is_tag_index) {
    if (max_path_load == 0) {
      if (is_residual) {
        init_with_seat_indices<dtype, false><<<grid_size, block_size, 0, stream>>>(
            0, dst_data_ptr, route_indices, loads, cell_num, cell_size, path_num, max_path_load);
      }
      if (gates_ptr == nullptr) {
        combine_with_seat_indices<dtype, false, false><<<grid_size, block_size, 0, stream>>>(
            src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
            path_num, max_path_load);
      } else {
        combine_with_seat_indices<dtype, true, false><<<grid_size, block_size, 0, stream>>>(
            src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
            path_num, max_path_load);
      }
    } else {
      if (is_residual) {
        init_with_seat_indices<dtype, true><<<grid_size, block_size, 0, stream>>>(
            0, dst_data_ptr, route_indices, loads, cell_num, cell_size, path_num, max_path_load);
      }
      if (gates_ptr == nullptr) {
        combine_with_seat_indices<dtype, false, true><<<grid_size, block_size, 0, stream>>>(
            src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
            path_num, max_path_load);
      } else {
        combine_with_seat_indices<dtype, true, true><<<grid_size, block_size, 0, stream>>>(
            src_data_ptr, dst_data_ptr, gates_ptr, route_indices, loads, cell_num, cell_size,
            path_num, max_path_load);
      }
    }
  } else {
    LOG(FATAL) << "Combine with tag indices is not supported yet.";
  }
}

// explicit instantiation

template void CombineWithSrcIndices<float>(void* src_data /*[total_loads, cell_size]*/,
                                           void* dst_data /*[cell_num, cell_size]*/,
                                           void* gates /*[cell_num, dst_num]*/,
                                           int* route_indices /*[cell_num, dst_num]*/,
                                           int* loads /*[dst_num]*/,
                                           const int& max_path_load,
                                           const int& cell_num,
                                           const int& cell_size,
                                           const int& path_num,
                                           cudaStream_t stream);

template void ResidualCombineWithSrcIndices<float>(void* src_data /*[total_loads, cell_size]*/,
                                                   void* dst_data /*[cell_num, cell_size]*/,
                                                   void* gates /*[cell_num x path_num]*/,
                                                   int* route_indices /*[cell_num x path_num]*/,
                                                   int* loads /*[path_num]*/,
                                                   const int& max_path_load,
                                                   const int& cell_num,
                                                   const int& cell_size,
                                                   const int& path_num,
                                                   cudaStream_t stream);

template void CombineWithSrcIndices<__half2>(void* src_data /*[total_loads, cell_size]*/,
                                             void* dst_data /*[cell_num, cell_size]*/,
                                             void* gates /*[cell_num, dst_num]*/,
                                             int* route_indices /*[cell_num, dst_num]*/,
                                             int* loads /*[dst_num]*/,
                                             const int& max_path_load,
                                             const int& cell_num,
                                             const int& cell_size,
                                             const int& path_num,
                                             cudaStream_t stream);

template void ResidualCombineWithSrcIndices<__half2>(void* src_data /*[total_loads, cell_size]*/,
                                                     void* dst_data /*[cell_num, cell_size]*/,
                                                     void* gates /*[cell_num, path_num]*/,
                                                     int* route_indices /*[cell_num, path_num]*/,
                                                     int* loads /*[path_num]*/,
                                                     const int& max_path_load,
                                                     const int& cell_num,
                                                     const int& cell_size,
                                                     const int& path_num,
                                                     cudaStream_t stream);

template void DispatchWithIndicesAndLoads<float>(void* src_data /*[cell_num, cell_size]*/,
                                                 void* dst_data /*[total_loads, cell_size]*/,
                                                 void* gates /*[cell_num, dst_num]*/,
                                                 int* route_indices /*[cell_num, dst_num]*/,
                                                 int* loads /*[dst_num]*/,
                                                 int* old_tags,
                                                 int* new_tags,
                                                 const int& cell_num,
                                                 const int& cell_size,
                                                 const int& path_num,
                                                 const int& max_path_load,
                                                 bool is_1d_routing,
                                                 bool is_tag_index,
                                                 cudaStream_t stream);

template void DispatchWithIndicesAndLoads<__half2>(void* src_data /*[cell_num, cell_size]*/,
                                                   void* dst_data /*[total_loads, cell_size]*/,
                                                   void* gates /*[cell_num, dst_num]*/,
                                                   int* route_indices /*[cell_num, dst_num]*/,
                                                   int* loads /*[dst_num]*/,
                                                   int* old_tags,
                                                   int* new_tags,
                                                   const int& cell_num,
                                                   const int& cell_size,
                                                   const int& path_num,
                                                   const int& max_path_load,
                                                   bool is_1d_routing,
                                                   bool is_tag_index,
                                                   cudaStream_t stream);
template void CombineWithIndicesAndLoads<float>(void* src_data /*[total_loads, cell_size]*/,
                                                void* dst_data /*[cell_num, cell_size]*/,
                                                void* gates /*[cell_num, dst_num]*/,
                                                int* route_indices /*[cell_num, dst_num]*/,
                                                int* loads /*[dst_num]*/,
                                                int* old_tags,
                                                int* new_tags,
                                                const int& cell_num,
                                                const int& cell_size,
                                                const int& path_num,
                                                const int& max_path_load,
                                                bool is_residual,
                                                bool is_tag_index,
                                                cudaStream_t stream);

template void CombineWithIndicesAndLoads<__half>(void* src_data /*[total_loads, cell_size]*/,
                                                 void* dst_data /*[cell_num, cell_size]*/,
                                                 void* gates /*[cell_num, dst_num]*/,
                                                 int* route_indices /*[cell_num, dst_num]*/,
                                                 int* loads /*[dst_num]*/,
                                                 int* old_tags,
                                                 int* new_tags,
                                                 const int& cell_num,
                                                 const int& cell_size,
                                                 const int& path_num,
                                                 const int& max_path_load,
                                                 bool is_residual,
                                                 bool is_tag_index,
                                                 cudaStream_t stream);
}  // namespace router
}  // namespace brt