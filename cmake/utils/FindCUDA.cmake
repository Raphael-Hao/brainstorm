# Licensed to the Apache Software Foundation (ASF) under one or more contributor
# license agreements.  See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.  The ASF licenses this
# file to you under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

# ##############################################################################
# Enhanced version of find CUDA.
#
# Usage: find_cuda(${USE_CUDA} ${USE_CUDNN})
#
# * When USE_CUDA=ON, use auto search
# * When USE_CUDA=/path/to/cuda-path, use the cuda path
# * When USE_CUDNN=ON, use auto search
# * When USE_CUDNN=/path/to/cudnn-path, use the cudnn path
#
# Provide variables:
#
# * CUDA_FOUND
# * CUDA_INCLUDE_DIRS
# * CUDA_TOOLKIT_ROOT_DIR
# * CUDA_CUDA_LIBRARY
# * CUDA_CUDART_LIBRARY
# * CUDA_NVRTC_LIBRARY
# * CUDA_CUDNN_INCLUDE_DIRS
# * CUDA_CUDNN_LIBRARY
# * CUDA_CUBLAS_LIBRARY
#
macro(find_cuda use_cuda use_cudnn use_nccl)
    set(__use_cuda ${use_cuda})
    if(${__use_cuda} MATCHES ${IS_TRUE_PATTERN})
        find_package(CUDA QUIET)
    elseif(IS_DIRECTORY ${__use_cuda})
        set(CUDA_TOOLKIT_ROOT_DIR ${__use_cuda})
        message(STATUS "Custom CUDA_PATH=" ${CUDA_TOOLKIT_ROOT_DIR})
        set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
        set(CUDA_FOUND TRUE)
        find_library(CUDA_CUDART_LIBRARY cudart ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                     ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    endif()

    # additional libraries
    if(CUDA_FOUND)
        find_library(
            _CUDA_CUDA_LIBRARY cuda
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib
                          targets/x86_64-linux/lib/stubs lib64/stubs
            NO_DEFAULT_PATH)
        if(_CUDA_CUDA_LIBRARY)
            set(CUDA_CUDA_LIBRARY ${_CUDA_CUDA_LIBRARY})
        endif()
        find_library(
            CUDA_NVRTC_LIBRARY nvrtc
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES
                lib lib64 targets/x86_64-linux/lib
                targets/x86_64-linux/lib/stubs lib64/stubs lib/x86_64-linux-gnu
            NO_DEFAULT_PATH)
        find_library(CUDA_CUBLAS_LIBRARY cublas ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                     ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)
        # search default path if cannot find cublas in non-default
        find_library(CUDA_CUBLAS_LIBRARY cublas)
        find_library(
            CUDA_CUBLASLT_LIBRARY
            NAMES cublaslt cublasLt
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib
            NO_DEFAULT_PATH)
        # search default path if cannot find cublaslt in non-default
        find_library(CUDA_CUBLASLT_LIBRARY NAMES cublaslt cublasLt)

        # find cuDNN
        set(__use_cudnn ${use_cudnn})
        if(${__use_cudnn} MATCHES ${IS_TRUE_PATTERN})
            set(CUDA_CUDNN_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
            find_library(
                CUDA_CUDNN_LIBRARY cudnn ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                ${CUDA_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)
            # search default path if cannot find cudnn in non-default
            find_library(CUDA_CUDNN_LIBRARY cudnn)
        elseif(IS_DIRECTORY ${__use_cudnn})
            # cuDNN doesn't necessarily live in the CUDA dir
            set(CUDA_CUDNN_ROOT_DIR ${__use_cudnn})
            set(CUDA_CUDNN_INCLUDE_DIRS ${CUDA_CUDNN_ROOT_DIR}/include)
            find_library(CUDA_CUDNN_LIBRARY cudnn ${CUDA_CUDNN_ROOT_DIR}/lib64
                         ${CUDA_CUDNN_ROOT_DIR}/lib NO_DEFAULT_PATH)
        endif()

        # find nccl
        set(__use_nccl ${use_nccl})
        set(CUDA_NCCL_INCLUDE_DIR
            $ENV{NCCL_INCLUDE_DIR}
            CACHE PATH "Folder contains NVIDIA NCCL headers")
        set(CUDA_NCCL_LIB_DIR
            $ENV{NCCL_LIB_DIR}
            CACHE PATH "Folder contains NVIDIA NCCL libraries")
        set(CUDA_NCCL_VERSION
            $ENV{NCCL_VERSION}
            CACHE STRING "Version of NCCL to build with")
        if(IS_DIRECTORY ${__use_nccl})
            set(CUDA_NCCL_ROOT_DIR ${__use_nccl})
            message(STATUS "Custom NCCL_PATH=" ${CUDA_NCCL_ROOT_DIR})
            set(CUDA_NCCL_INCLUDE_DIR ${CUDA_NCCL_ROOT_DIR}/include)
            set(CUDA_NCCL_LIB_DIR ${CUDA_NCCL_ROOT_DIR}/lib)
        endif()
        list(APPEND CUDA_NCCL_ROOT_DIR $ENV{NCCL_ROOT_DIR}
             ${CUDA_TOOLKIT_ROOT_DIR})
        # Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for
        # searching paths and libraries for CMake >=3.12.
        # list(APPEND CMAKE_PREFIX_PATH ${CUDA_NCCL_ROOT_DIR})

        find_path(
            CUDA_NCCL_INCLUDE_DIRS
            NAMES nccl.h
            HINTS ${CUDA_NCCL_INCLUDE_DIR})

        if(USE_STATIC_NCCL)
            message(
                STATUS
                    "USE_STATIC_NCCL is set. Linking with static NCCL library.")
            set(NCCL_LIBNAME "nccl_static")
            if(NCCL_VERSION) # Prefer the versioned library if a specific NCCL
                             # version is specified
                set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}"
                                                ${CMAKE_FIND_LIBRARY_SUFFIXES})
            endif()
        else()
            set(NCCL_LIBNAME "nccl")
            if(NCCL_VERSION) # Prefer the versioned library if a specific NCCL
                             # version is specified
                set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}"
                                                ${CMAKE_FIND_LIBRARY_SUFFIXES})
            endif()
        endif(USE_STATIC_NCCL)

        find_library(
            CUDA_NCCL_LIBRARY
            NAMES ${NCCL_LIBNAME}
            HINTS ${CUDA_NCCL_LIB_DIR})

        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(
            NCCL DEFAULT_MSG CUDA_NCCL_INCLUDE_DIRS CUDA_NCCL_LIBRARY)

        message(STATUS "Found CUDA_TOOLKIT_ROOT_DIR=" ${CUDA_TOOLKIT_ROOT_DIR})
        message(STATUS "Found CUDA_CUDA_LIBRARY=" ${CUDA_CUDA_LIBRARY})
        message(STATUS "Found CUDA_CUDART_LIBRARY=" ${CUDA_CUDART_LIBRARY})
        message(STATUS "Found CUDA_NVRTC_LIBRARY=" ${CUDA_NVRTC_LIBRARY})
        message(STATUS "Found CUDA_CUDNN_INCLUDE_DIRS="
                       ${CUDA_CUDNN_INCLUDE_DIRS})
        message(STATUS "Found CUDA_CUDNN_LIBRARY=" ${CUDA_CUDNN_LIBRARY})
        message(STATUS "Found CUDA_CUBLAS_LIBRARY=" ${CUDA_CUBLAS_LIBRARY})
        message(STATUS "Found CUDA_CUBLASLT_LIBRARY=" ${CUDA_CUBLASLT_LIBRARY})
        message(STATUS "Found CUDA_NCCL_INCLUDE_DIRS="
                       ${CUDA_NCCL_INCLUDE_DIRS})
        message(STATUS "Found CUDA_NCCL_LIBRARY=" ${CUDA_NCCL_LIBRARY})

    endif(CUDA_FOUND)
endmacro(find_cuda)
