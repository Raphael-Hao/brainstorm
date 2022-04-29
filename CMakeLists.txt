cmake_minimum_required(VERSION 3.2)
project(brainstorm LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/FindCUDA.cmake)
include(cmake/Utils.cmake)

include(cmake/brainstorm.cmake)
include(cmake/tvm.cmake)
include(cmake/nnfusion.cmake)

brt_option(PTX_INFO "print ptx info for each cuda kernel" OFF)

if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#   SET(CMAKE_INSTALL_PREFIX  CACHE PATH <comment> FORCE)
# ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# thirdparty libraries
add_subdirectory(3rdparty/tvm)

#include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(include)
include_directories(3rdparty/tvm/include)
include_directories(3rdparty/dmlc-core/include)
include_directories(3rdparty/dlpack/include)

set(IS_FALSE_PATTERN "^[Oo][Ff][Ff]$|^0$|^[Ff][Aa][Ll][Ss][Ee]$|^[Nn][Oo]$|^[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|.*-[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|^$")
set(IS_TRUE_PATTERN "^[Oo][Nn]$|^[1-9][0-9]*$|^[Tt][Rr][Uu][Ee]$|^[Yy][Ee][Ss]$|^[Yy]$")
set(BRT_LINKER_LIBS "")

set(WARNING_FLAG -Wall)
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "Build in Debug mode")
  set(CMAKE_C_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
else()
  set(CMAKE_C_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(BRT_VISIBILITY_FLAG "")
  if (HIDE_PRIVATE_SYMBOLS)
    message(STATUS "Hide private symbols...")
    set(BRT_VISIBILITY_FLAG "-fvisibility=hidden")
  endif(HIDE_PRIVATE_SYMBOLS)
endif ()
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
endif()

list(APPEND BRT_LINKER_LIBS ${CMAKE_DL_LIBS})

set(CMAKE_CXX_FLAGS "/std:c++14 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD 14)

find_cuda(ON ON)
if (CUDA_FOUND)
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

if(PTX_INFO)
  message(STATUS "Build with PTX info")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
endif(PTX_INFO)


# add_subdirectory(microbench)
add_subdirectory(benchmark)
add_subdirectory(test)
