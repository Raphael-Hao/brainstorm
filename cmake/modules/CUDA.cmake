
set(USE_CUDA ON)
set(USE_CUDNN ON)
set(USE_CUBLAS ON)
set(USE_NCCL ON)
find_cuda(ON ON ON)
if(CUDA_FOUND)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()
if(NOT CUDA_FOUND)
  message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
endif()
message(STATUS "Build with CUDA ${CUDA_VERSION} support")
list(APPEND BRT_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
list(APPEND BRT_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
list(APPEND BRT_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
if(USE_CUDNN)
  message(STATUS "Build with cuDNN support")
  include_directories(SYSTEM ${CUDA_CUDNN_INCLUDE_DIRS})
  list(APPEND BRT_LINKER_LIBS ${CUDA_CUDNN_LIBRARY})
endif(USE_CUDNN)

if(USE_CUBLAS)
  message(STATUS "Build with cuBLAS support")
  list(APPEND BRT_LINKER_LIBS ${CUDA_CUBLAS_LIBRARY})
  if(NOT CUDA_CUBLASLT_LIBRARY STREQUAL "CUDA_CUBLASLT_LIBRARY-NOTFOUND")
    list(APPEND BRT_LINKER_LIBS ${CUDA_CUBLASLT_LIBRARY})
  endif()
endif(USE_CUBLAS)

if(USE_THRUST)
  message(STATUS "Build with Thrust support")
  cmake_minimum_required(VERSION 3.13) # to compile CUDA code
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
endif(USE_THRUST)

if(USE_NCCL)
  message(STATUS "Build with NCCL support")
    include_directories(SYSTEM ${CUDA_NCCL_INCLUDE_DIRS})
  list(APPEND BRT_LINKER_LIBS ${CUDA_NCCL_LIBRARY})
endif(USE_NCCL)
if(PTX_INFO)
  message(STATUS "Build with PTX info")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
endif(PTX_INFO)


if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()