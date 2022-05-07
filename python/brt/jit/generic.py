# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import re

from brt.common import log

from .utils import remove_empty_lines

logger = log.get_logger(__file__)

CUDATypeSizeInByte = {
    # signed type
    "char": 1,
    "short": 2,
    "int": 4,
    "float": 4,
    "double": 8,
    "int8_t": 1,
    "int16_t": 2,
    "int32_t": 4,
    "int64_t": 8,
    # unsigned type
    "uchar": 1,
    "ushort": 2,
    "uint": 4,
    "uint8_t": 1,
    "uint16_t": 2,
    "uint32_t": 4,
    "uint64_t": 8,
}


class GenericFunction:
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
  asm volatile("bar.warp.sync %0;" : : "r"(threadsmask), : "memory");
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
            self.max_threads_per_block = None
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
        self.launch_bounds = int(parsed_source.group(1))
        self.name = parsed_source.group(2)
        self.args = parsed_source.group(3)
        self.body = parsed_source.group(4)
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

    def clean_cache_and_set_mode(self, mode="global"):
        self.mode = mode
        self.clean_code = ""
        self.indent = 0

    def new_codeblock(self):
        self.clean_code += "{\n"
        self.indent += 1

    def close_codeblock(self):
        self.clean_code += "{\n"
        self.indent -= 1

    def set_launch_bounds(self):
        if self.max_threads_per_block is None:
            return
        if self.min_blocks_per_sm == 1:
            self.clean_code += f"__launch_bounds__({self.max_threads_per_block})"
        else:
            self.clean_code += f"__launch_bounds__({self.max_threads_per_block}, {self.min_blocks_per_sm})"

    def declare_name_args(self):
        self.clean_code += f" {self.name}("
        self.clean_code += f"{self.args}"
        if self.mode == "device":
            self.clean_code += (
                f", char* shared_buffer, const uint& block_idx, const uint& thread_idx"
            )
        self.clean_code += ")"

    def set_culaunch_dims(self):
        if self.mode == "device":
            self.clean_code += f"  // [thread_extent] gridSize = {self.grid_size}\n"
            self.clean_code += f"  // [thread_extent] blockSize = {self.block_size}\n"
        else:
            self.clean_code += (
                f"  // [thread_extent] blockIdx.xdim = {self.blockidx_xdim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] blockIdx.ydim = {self.blockidx_ydim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] blockIdx.zdim = {self.blockidx_zdim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] threadIdx.xdim = {self.threadidx_xdim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] threadIdx.ydim = {self.threadidx_ydim}\n"
            )
            self.clean_code += (
                f"  // [thread_extent] threadIdx.zdim = {self.threadidx_zdim}\n"
            )
        self.clean_code += "\n"

    def alloc_shared_memory(self):
        if self.shm_size_in_bytes > 0:
            if self.mode == "device":
                allocated_shm_size = 0
                for i in range(len(self.shm_sizes)):
                    self.clean_code += f"  {self.shm_types[i]}* {self.shm_symbols[i]} = ({self.shm_types[i]}*)(shared_buffer + {allocated_shm_size});\n"
                    allocated_shm_size += (
                        self.shm_sizes[i] * CUDATypeSizeInByte[self.shm_types[i]]
                    )
                assert allocated_shm_size == self.shm_size_in_bytes
            else:
                for i in range(len(self.shm_sizes)):
                    self.clean_code += f"  __shared__ {self.shm_types[i]} {self.shm_symbols[i]}[{self.shm_sizes[i]}];\n"

    def verify_code(self):
        try:
            assert self.indent == 0
        except AssertionError:
            logger.exception("Code verify failed")

    def get_code(self, mode: str = "global") -> str:
        self.clean_cache_and_set_mode(mode)
        if self.mode == "global":
            self.clean_code += GenericFunction.common_defines
            self.clean_code += GenericFunction.c_api_decorator
            self.clean_code += GenericFunction.global_decorator
            self.declare_name_args()
            self.set_launch_bounds()
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
            self.clean_code += (
                f"  uint block_idx_x = block_idx % {self.blockidx_xdim};\n"
            )
            self.clean_code += f"  uint block_idx_y = (block_idx / {self.blockidx_xdim}) % {self.blockidx_ydim};\n"
            self.clean_code += (
                f"  uint block_idx_z = block_idx / {self.blockidx_xydim};\n"
            )
            self.clean_code += (
                f"  uint thread_idx_x = thread_idx % {self.threadidx_xdim};\n"
            )
            self.clean_code += f"  uint thread_idx_y = (thread_idx / {self.threadidx_xdim}) % {self.threadidx_ydim};\n"
            self.clean_code += (
                f"  uint thread_idx_z = thread_idx / {self.threadidx_xydim};\n"
            )
            self.clean_code += (
                f"  dim3 brt_blockIdx(block_idx_x, block_idx_y, block_idx_z);\n"
            )
            self.clean_code += (
                f"  dim3 brt_threadIdx(thread_idx_x, thread_idx_y, thread_idx_z);\n"
            )
            self.alloc_shared_memory()

            device_func_body = self.body.replace("blockIdx", "brt_blockIdx")
            device_func_body = device_func_body.replace("threadIdx", "brt_threadIdx")
            self.clean_code += f"{device_func_body}"
            self.close_codeblock()
        else:
            raise ValueError("Invalid mode: %s" % mode)
        return self.clean_code

