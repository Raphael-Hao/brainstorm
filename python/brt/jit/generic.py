# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import re

from brt.common import log

from .base import BaseFunction, CUDATypeSizeInByte
from .utils import remove_empty_lines

logger = log.get_logger(__file__)


class GenericFunction(BaseFunction):
    common_defines = """
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
"""
    asm_block_sync = """
__device__ __forceinline__ void AsmBlockSync(int name, int numThreads) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}
"""
    asm_warp_sync = """
__device__ __forceinline__ void AsmWarpSync(const unsigned threadsmask) {
  asm volatile("bar.warp.sync %0;" : : "r"(threadsmask) : "memory");
}
"""
    cpp_warp_sync = """
__device__ __forceinline__ void CppWarpSync(const unsigned threadsmask) {
  __syncwarp(threadsmask);
}   
"""
    cpp_cg_warp_sync = """
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__device__ __forceinline__ void CppCgWarpSync() {
  cg::coalesced_group g = cg::coalesced_threads();
  g.sync();
}
"""
    c_api_decorator = 'extern "C" '
    global_decorator = "__global__ void "
    device_decorator = "__device__ __forceinline__ void "

    def __init__(self, raw_source) -> None:
        self.raw_source = raw_source
        self.extract_raw_source()
        self.extract_culaunch_dims()
        self.extract_func_args()
        self.extract_shared_memory()
        self.clean_raw_body()
        self.calcu_sync_mask()

    def extract_raw_source(self):
        """
        Parse raw source code to extract function name and arguments.
        """
        # Example:
        # extern "C" __global__ void __launch_bounds__(32) fuse_add_blocks(float* %0, float* %1, float* %2 ) {}
        launch_bound_regex = r"\s+__launch_bounds__\((\w+)\)\s+"
        launch_bounds = re.findall(launch_bound_regex, self.raw_source)
        self.min_blocks_per_sm = 1
        if len(launch_bounds) == 0:
            self.max_threads_per_block = 0
        else:
            launch_bound_params = launch_bounds[0].split(",")
            self.max_threads_per_block = int(launch_bound_params[0])
            if len(launch_bound_params) == 2:
                self.min_blocks_per_sm = int(launch_bound_params[1])
            source_without_launch_bound = re.sub(
                launch_bound_regex, " ", self.raw_source
            )
        parsed_source = re.search(
            r"extern\s+\"C\"\s+__global__\s+void\s+(\w+)\s*\((.*)\)\s*(\{[\s\S]*)",
            source_without_launch_bound,
        )
        self.name = parsed_source.group(1)
        self.args = parsed_source.group(2)
        self.body = parsed_source.group(3)
        self.body = self.body[self.body.find("{") + 1 : self.body.rfind("}")]

    def extract_culaunch_dims(self):
        self.blockidx_xdim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.xdim\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.blockidx_ydim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.ydim\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.blockidx_zdim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*blockIdx.zdim\s*=\s*(\d+)", self.raw_source
            ).group(1)
        )
        self.threadidx_xdim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.xdim\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.threadidx_ydim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.ydim\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.threadidx_zdim = int(
            re.search(
                r"\/\/\s*\[thread_extent\]\s*threadIdx.zdim\s*=\s*(\d+)",
                self.raw_source,
            ).group(1)
        )
        self.blockidx_xydim = self.blockidx_xdim * self.blockidx_ydim
        self.threadidx_xydim = self.threadidx_xdim * self.threadidx_ydim
        self.grid_size = self.blockidx_xydim * self.blockidx_zdim
        self.block_size = self.threadidx_xydim * self.threadidx_zdim

    def extract_func_args(self):
        # find all function arguments
        self.arg_types = []
        self.arg_decorators = []
        self.arg_names = []
        for func_arg in self.args.split(","):
            func_arg = func_arg.strip()
            func_arg_info = func_arg.split(" ")
            if len(func_arg_info) == 2:
                self.arg_types.append(func_arg_info[0])
                self.arg_decorators.append([""])
                self.arg_names.append(func_arg_info[1])
            elif len(func_arg_info) == 3:
                self.arg_types.append(func_arg_info[0])
                self.arg_decorators.append(func_arg_info[1])
                self.arg_names.append(func_arg_info[2])
            else:
                raise ValueError("Invalid function argument: %s" % func_arg)

    def extract_shared_memory(self):
        # find all shared memory declarations
        shm_regex = r"__shared__\s+(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;"
        shm_declares = re.findall(shm_regex, self.body)
        if len(shm_declares) == 0:
            self.shm_size_in_bytes = 0
            return
        self.shm_types = []
        self.shm_symbols = []
        self.shm_sizes = []
        self.shm_size_in_bytes = 0
        for shm_declare in shm_declares:
            shm_type = shm_declare[0]
            shm_symbol = shm_declare[1]
            shm_size = int(shm_declare[2])
            self.shm_types.append(shm_type)
            self.shm_symbols.append(shm_symbol)
            self.shm_sizes.append(shm_size)
            self.shm_size_in_bytes += shm_size * CUDATypeSizeInByte[shm_type]
        self.body = re.sub(shm_regex, "", self.body)

    def clean_raw_body(self):
        self.body = remove_empty_lines(self.body)

    def extract_syncthreads_times(self):
        # only count the number of __syncthreads existing in the code
        self.syncthreads_times = self.body.count("__syncthreads()")

    def calcu_sync_mask(self):
        if self.block_size < 32:
            sync_mask_bin_str = "1" * self.block_size
            sync_mask_int = int(sync_mask_bin_str, 2) << (32 - self.block_size)
            self.sync_mask = f"{hex(sync_mask_int)}"
        else:
            self.sync_mask = "0xffffffff"

    # Construct Generic Functions

    def shadow_global_dims(self):
        self.clean_code += f"  const dim3 gridDim({self.blockidx_xdim}, {self.blockidx_ydim}, {self.blockidx_zdim});\n"
        self.clean_code += f"  const dim3 blockDim({self.threadidx_xdim}, {self.threadidx_ydim}, {self.threadidx_zdim});\n"
        self.clean_code += "\n"
        if self.threadidx_ydim != 1 and self.threadidx_zdim == 1:
            self.clean_code += f"  const dim3 threadIdx(thread_idx % {self.threadidx_xdim}, thread_idx / {self.threadidx_xdim});\n"
        elif self.threadidx_ydim == 1 and self.threadidx_zdim != 1:
            self.clean_code += f"  const dim3 threadIdx(thread_idx % {self.threadidx_xdim}, 1, thread_idx / {self.threadidx_xdim});\n"
        elif self.threadidx_ydim != 1 and self.threadidx_zdim != 1:
            self.clean_code += f"  const dim3 threadIdx(thread_idx % {self.threadidx_xdim}, thread_idx / {self.threadidx_xdim} % {self.threadidx_ydim}, thread_idx / {self.threadidx_xydim});\n"
        if self.blockidx_ydim == 1 and self.blockidx_zdim == 1:
            self.clean_code += f"  const dim3 blockIdx(block_idx);\n"
        elif self.blockidx_zdim == 1:
            self.clean_code += f"  const dim3 blockIdx(block_idx % {self.blockidx_xdim}, block_idx % {self.blockidx_xdim});\n"
        else:
            self.clean_code += f"  const dim3 blockIdx(block_idx % {self.blockidx_xdim}, block_idx / {self.blockidx_xdim} % {self.blockidx_ydim}, block_idx / {self.blockidx_xydim});\n"

    def add_body_without_syncthreads(self, device_id: int, sync_method="asm"):
        if sync_method == "asm":
            if self.block_size >= 32:
                body_without_syncthreads = self.body.replace(
                    "__syncthreads()", f"AsmBlockSync({device_id}, {self.block_size})"
                )
            else:
                body_without_syncthreads = self.body.replace(
                    "__syncthreads()", f"AsmWarpSync({self.sync_mask})"
                )
        elif sync_method == "cpp":
            if self.block_size > 32:
                logger.error("CPP sync is not supported for block size >= 32")
            else:
                body_without_syncthreads = self.body.replace(
                    "__syncthreads()", f"CppWarpSync({self.sync_mask})"
                )
        else:
            logger.error("Unknown sync method")
        self.clean_code += body_without_syncthreads

    def get_code(self, mode="global", device_id: int = 0, sync_method="asm") -> str:
        self.reset_mode(mode)
        if self.mode == "global":
            self.clean_code += GenericFunction.common_defines
            self.clean_code += GenericFunction.c_api_decorator
            self.clean_code += GenericFunction.global_decorator
            self.set_launch_bounds()
            self.declare_name_args()
            self.new_codeblock()
            self.set_culaunch_dims()
            self.alloc_shared_memory()
            self.clean_code += f"{self.body}"
            self.close_codeblock()
        elif self.mode == "device":
            self.clean_code += GenericFunction.device_decorator
            self.declare_name_args()
            self.new_codeblock()
            self.set_culaunch_dims()
            self.clean_code += f"  if (thread_idx >= {self.block_size})"
            self.new_codeblock()
            self.clean_code += "    return;\n"
            self.close_codeblock()
            self.shadow_global_dims()
            self.alloc_shared_memory()
            self.add_body_without_syncthreads(device_id, sync_method)
            self.close_codeblock()
        else:
            raise ValueError("Invalid mode: %s" % mode)
        self.verify_code()
        return self.clean_code

