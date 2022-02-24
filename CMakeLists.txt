cmake_minimum_required(VERSION 3.2)
project(brainstorm C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(cmake/brainstorm.cmake)
include(cmake/tvm.cmake)
include(cmake/nnfusion.cmake)

# IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#   SET(CMAKE_INSTALL_PREFIX  CACHE PATH <comment> FORCE)
# ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# thirdparty libraries
add_subdirectory(3rdparty/tvm)
# add_subdirectory(3rdparty/nnfusion)

#include directories
include_directories(include)
include_directories(3rdparty/tvm/include)
include_directories(3rdparty/tvm/3rdparty/dmlc-core/include)
include_directories(3rdparty/tvm/3rdparty/dlpack/include)

add_subdirectory(microbench)
