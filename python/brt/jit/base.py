# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from brt.common import log

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


class Function:
    c_api_decorator = 'extern "C" {\n'
    def __init__(self) -> None:
        pass
    
    def add_c_api(self):
        self.clean_code += Function.c_api_decorator
    
    def end_c_api(self):
        self.clean_code += "} // extern \"C\"\n"
    
    def add_codeblock(self, codeblock: str):
        self.clean_code += codeblock
        self.new_emtpy_line()
        
    def add_line_with_indent(self, code: str, end=False):
        self.clean_code += "  " * self.indent
        self.clean_code += code
        if end == True:
            self.end_line()

    def add_code(self, code: str, end=False):
        self.clean_code += code
        if end == True:
            self.end_line()

    def end_line(self):
        self.clean_code += ";\n"

    def new_emtpy_line(self):
        self.clean_code += "\n"

    def new_codeblock(self):
        self.clean_code += "{\n"
        self.indent += 1

    def close_codeblock(self):
        self.indent -= 1
        self.clean_code += "  " * self.indent
        self.clean_code += "}\n"

    def verify_code(self):
        try:
            assert self.indent == 0
        except AssertionError:
            logger.exception("Code verify failed")


class GlobalFunction(Function):
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

    def __init__(self) -> None:
        pass

    def set_launch_bounds(self):
        if self.max_threads_per_block == 0:
            return
        if self.min_blocks_per_sm == 1:
            self.clean_code += f"__launch_bounds__({self.max_threads_per_block}) "
        else:
            self.clean_code += f"__launch_bounds__({self.max_threads_per_block}, {self.min_blocks_per_sm}) "

    def declare_name_args(self):
        self.clean_code += f"{self.name}("
        self.clean_code += f"{self.args}"
        if self.mode == "device":
            self.clean_code += (
                f", char* shared_buffer, const uint& block_idx, const uint& thread_idx"
            )
        self.clean_code += ") "

    def set_kernel_type(self, kernel_type: str = "global"):
        if self.mode == "device":
            logger.error("Kernel type not supported in device mode")
        self.clean_code += f"  // [kernel_type] {kernel_type}\n"

    def set_culaunch_dims(self):
        if self.mode == "global":
            # self.clean_code += f"  // [thread_extent] gridSize = {self.grid_size}\n"
            # self.clean_code += f"  // [thread_extent] blockSize = {self.block_size}\n"
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
        else:
            logger.error("Culaunch dims not supported in device mode")
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

    @property
    def sync_mask(self):
        if self.block_size < 32:
            sync_mask_bin_str = "1" * self.block_size
            sync_mask_int = int(sync_mask_bin_str, 2) << (32 - self.block_size)
            return f"{hex(sync_mask_int)}"
        else:
            return "0xffffffff"

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

    def reset_mode(self, mode="global"):
        self.mode = mode
        self.clean_code = ""
        self.indent = 0

    def get_code(self, mode="global", device_id: int = 0, sync_method="asm"):
        self.reset_mode(mode)
        if self.mode == "global":
            self.clean_code += GlobalFunction.common_defines
            self.clean_code += GlobalFunction.c_api_decorator
            self.clean_code += GlobalFunction.global_decorator
            self.set_launch_bounds()
            self.declare_name_args()
            self.new_codeblock()
            self.set_kernel_type("global")
            self.set_culaunch_dims()
            self.alloc_shared_memory()
            self.clean_code += f"{self.body}"
            self.close_codeblock()
        elif self.mode == "device":
            self.clean_code += GlobalFunction.device_decorator
            self.declare_name_args()
            self.new_codeblock()
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

