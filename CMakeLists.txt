cmake_minimum_required(VERSION 3.13.0)
project(brainstorm LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/utils/Utils.cmake)
include(cmake/utils/FindCUDA.cmake)

include(cmake/config/brainstorm.cmake)
# include(cmake/config/tvm.cmake) include(cmake/config/nnfusion.cmake)

brt_option(PTX_INFO "print ptx info for each cuda kernel" OFF)

include(cmake/modules/CUDA.cmake)

# IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) SET(CMAKE_INSTALL_PREFIX CACHE
# PATH <comment> FORCE) ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# thirdparty libraries
# add_subdirectory(3rdparty/tvm)

# include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(include)
include_directories(SYSTEM 3rdparty/dmlc-core/include)
include_directories(SYSTEM 3rdparty/dlpack/include)
# include_directories(SYSTEM 3rdparty/tvm/include)

set(BRT_LINKER_LIBS "")
set(WARNING_FLAG -Wall)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS
        "-O0 -g -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
else()
    set(CMAKE_C_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS
        "-O2 -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
    set(BRT_VISIBILITY_FLAG "")
    if(HIDE_PRIVATE_SYMBOLS)
        message(STATUS "Hide private symbols...")
        set(BRT_VISIBILITY_FLAG "-fvisibility=hidden")
    endif(HIDE_PRIVATE_SYMBOLS)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                           VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
endif()

list(APPEND BRT_LINKER_LIBS ${CMAKE_DL_LIBS})

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD 14)

file(GLOB_RECURSE BRT_SOURCE CMAKE_CONFIGURE_DEPENDS "src/*.cc" "src/*.cu")
# remove srcs of extension for other frameworks, they will be compiled by python
set(EXCLUDE_DIR "jit/extension/")
foreach(TMP_SRC ${BRT_SOURCE})
    string(FIND ${TMP_SRC} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
    if(NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
        message(STATUS "removing ${TMP_SRC}")
        list(REMOVE_ITEM BRT_SOURCE ${TMP_SRC})
    endif()
endforeach(TMP_SRC)

file(GLOB_RECURSE BRT_INCLUDE CMAKE_CONFIGURE_DEPENDS "src/*.h" "include/*.h")

list(APPEND BRT_SRCS ${BRT_SOURCE} ${BRT_INCLUDE})


add_compile_definitions("USE_CUDA")
add_library(brt SHARED ${BRT_SRCS})
target_include_directories(
    brt PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(brt PUBLIC ${BRT_LINKER_LIBS})
# set_target_properties(brt PROPERTIES LINKER_LANGUAGE CXX)

add_subdirectory(benchmark)
add_subdirectory(test)
