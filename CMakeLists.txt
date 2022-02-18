cmake_minimum_required(VERSION 3.2)
project(brainstorm C CXX)
include(cmake/tvm_config.cmake)

# thirdparty libraries
add_subdirectory(3rdparty/tvm)
add_subdirectory(3rdparty/nnfusion)

#include directories
include_directories(3rdparty/tvm/include)
include_directories(3rdparty/tvm/3rdparty/dmlc-core/include)
include_directories(3rdparty/tvm/3rdparty/dlpack/include)
