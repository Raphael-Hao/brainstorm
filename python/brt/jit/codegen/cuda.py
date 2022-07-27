# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.runtime import log
from brt.jit import CUDACompiler

from brt.jit.codegen.kernel import Kernel

logger = log.get_logger(__file__)


__all__ = ["CUDATypeSizeInByte", "GlobalKernel"]

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


class GlobalKernel(Kernel):
    common_defines = """
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
"""
    asm_block_sync = """
extern "C" __device__ __forceinline__ void AsmBlockSync(int name, int block_size) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(block_size) : "memory");
}
"""
    asm_warp_sync = """
extern "C" __device__ __forceinline__ void AsmWarpSync(const unsigned threads_mask) {
  asm volatile("bar.warp.sync %0;" : : "r"(threads_mask) : "memory");
}
"""
    cpp_warp_sync = """
extern "C" __device__ __forceinline__ void CppWarpSync(const unsigned threads_mask) {
  __syncwarp(threads_mask);
}   
"""
    cpp_cg_block_sync = """
#include <cooperative_groups.h>
extern "C" __device__ __forceinline__ void CppCgBlockSync(int block_size) {
  thread_group real_blocks = cooperative_groups::partition(this_thread_block(), block_size);
  real_blocks.sync();
}
"""
    c_api_decorator = 'extern "C" '
    global_decorator = "__global__ "
    device_decorator = "__device__ __forceinline__ "

    def __init__(self) -> None:
        if not hasattr(self, "kernel_type"):
            setattr(self, "kernel_type", "global")
        self.initialized = False

    def declare_return_with_launch_bounds(self):
        formated_code = "void "
        if self.max_threads_per_block == 0 or self.mode == "device":
            self.append_code(formated_code)
            return formated_code
        if self.min_blocks_per_sm == 1:
            formated_code += f"__launch_bounds__({self.max_threads_per_block}) "
        else:
            formated_code += f"__launch_bounds__({self.max_threads_per_block}, {self.min_blocks_per_sm}) "
        self.append_code(formated_code)
        return formated_code

    def declare_name_args(self):
        formated_code = f"{self.func_name}("
        formated_code += f"{self.args}"
        if self.mode == "device":
            formated_code += (
                f", char* shared_buffer, const uint& block_idx, const uint& thread_idx"
            )
        formated_code += ") "
        self.append_code(formated_code)
        return formated_code

    def set_kernel_type(self):
        if self.mode == "device":
            logger.error("Kernel type not supported in device mode")
        formated_code = self.add_line_with_indent(
            f"// [kernel_type] {self.kernel_type}", end=True
        )
        return formated_code

    def set_culaunch_dims(self):
        if self.mode == "global":
            # self.clean_code += f"  // [thread_extent] gridSize = {self.grid_size}\n"
            # self.clean_code += f"  // [thread_extent] blockSize = {self.block_size}\n"

            formated_code = self.add_line_with_indent(
                f"// [thread_extent] blockIdx.x = {self.blockidx_x}", end=True
            )
            formated_code += self.add_line_with_indent(
                f"// [thread_extent] blockIdx.y = {self.blockidx_y}", end=True
            )
            formated_code += self.add_line_with_indent(
                f"// [thread_extent] blockIdx.z = {self.blockidx_z}", end=True
            )
            formated_code += self.add_line_with_indent(
                f"// [thread_extent] threadIdx.x = {self.threadidx_x}", end=True
            )
            formated_code += self.add_line_with_indent(
                f"// [thread_extent] threadIdx.y = {self.threadidx_y}", end=True
            )
            formated_code += self.add_line_with_indent(
                f"// [thread_extent] threadIdx.z = {self.threadidx_z}", end=True
            )
            formated_code += self.new_line()
            return formated_code
        else:
            logger.error("Culaunch dims not supported in device mode")

    def alloc_shared_memory(self):
        formated_code = ""
        if self.shm_size_in_bytes > 0:
            if self.mode == "device":
                allocated_shm_size = 0
                for i in range(len(self.shm_sizes)):
                    formated_code += self.add_line_with_indent(
                        f"{self.shm_types[i]}* {self.shm_symbols[i]} = ({self.shm_types[i]}*)(shared_buffer + {allocated_shm_size});",
                        end=True,
                    )
                    allocated_shm_size += (
                        self.shm_sizes[i] * CUDATypeSizeInByte[self.shm_types[i]]
                    )
                assert allocated_shm_size == self.shm_size_in_bytes
            else:
                for i in range(len(self.shm_sizes)):
                    formated_code += self.add_line_with_indent(
                        f"__shared__ {self.shm_types[i]} {self.shm_symbols[i]}[{self.shm_sizes[i]}];",
                        end=True,
                    )
        return formated_code

    @property
    def sync_mask(self):
        if self.block_size < 32:
            sync_mask_bin_str = "1" * self.block_size
            sync_mask_int = int(sync_mask_bin_str, 2) << (32 - self.block_size)
            return f"{hex(sync_mask_int)}"
        else:
            return "0xffffffff"

    def shadow_global_dims(self):
        formated_code = self.add_line_with_indent(
            f"const dim3 gridDim({self.blockidx_x}, {self.blockidx_y}, {self.blockidx_z});",
            end=True,
        )
        formated_code += self.add_line_with_indent(
            f"const dim3 blockDim({self.threadidx_x}, {self.threadidx_y}, {self.threadidx_z});",
            end=True,
        )
        formated_code += self.new_line()
        if self.threadidx_y != 1 and self.threadidx_z == 1:
            formated_code += self.add_line_with_indent(
                f"const dim3 threadIdx(thread_idx % {self.threadidx_x}, thread_idx / {self.threadidx_x});",
                end=True,
            )
        elif self.threadidx_y == 1 and self.threadidx_z != 1:
            formated_code += self.add_line_with_indent(
                f"const dim3 threadIdx(thread_idx % {self.threadidx_x}, 1, thread_idx / {self.threadidx_x});",
                end=True,
            )
        elif self.threadidx_y != 1 and self.threadidx_z != 1:
            formated_code += self.add_line_with_indent(
                f"const dim3 threadIdx(thread_idx % {self.threadidx_x}, thread_idx / {self.threadidx_x} % {self.threadidx_y}, thread_idx / {self.threadidx_xydim});",
                end=True,
            )
        if self.blockidx_y == 1 and self.blockidx_z == 1:
            formated_code += self.add_line_with_indent(
                f"const dim3 blockIdx(block_idx);", end=True
            )
        elif self.blockidx_z == 1:
            formated_code += self.add_line_with_indent(
                "const dim3 blockIdx(block_idx % {self.blockidx_x}, block_idx % {self.blockidx_x});",
                end=True,
            )
        else:
            formated_code += self.add_line_with_indent(
                f"const dim3 blockIdx(block_idx % {self.blockidx_x}, block_idx / {self.blockidx_x} % {self.blockidx_y}, block_idx / {self.blockidx_xydim});",
                end=True,
            )
        return formated_code

    def add_body_without_syncthreads(self):
        if self.block_size >= 32:
            real_block_size = self.block_size
            if real_block_size % 32 != 0:
                real_block_size = 32 * (real_block_size // 32 + 1)
            body_without_syncthreads = self.raw_body.replace(
                "__syncthreads()", f"AsmBlockSync(0, {real_block_size})"
            )
        else:
            body_without_syncthreads = self.raw_body.replace(
                "__syncthreads()", f"AsmWarpSync({self.sync_mask})"
            )
        formated_code = self.add_codeblock(body_without_syncthreads)
        return formated_code

    def reset(self, mode):
        self.mode = mode
        self.clean_code = ""
        self.indent = 0

    def generate_dependency(self, sync_method="asm"):
        deps = []
        return deps

    def generate_signature(self):
        formated_code = self.add_line_with_indent(GlobalKernel.global_decorator)
        formated_code += self.declare_return_with_launch_bounds()
        formated_code += self.declare_name_args()
        return formated_code

    def generate_body(self):
        formated_code = self.new_codeblock()
        formated_code += self.set_kernel_type()
        formated_code += self.set_culaunch_dims()
        formated_code += self.alloc_shared_memory()
        formated_code += self.add_codeblock(self.raw_body)
        formated_code += self.close_codeblock()
        return formated_code

    def get_code(self):
        assert self.initialized is True, "CodeGenerator is not initialized"
        self.reset(mode="global")
        self.add_codeblock(GlobalKernel.common_defines)
        if (
            hasattr(self, "func_deps")
            and hasattr(self, "func_sig")
            and hasattr(self, "func_body")
        ):
            func_deps = getattr(self, "func_deps")
            func_sig = getattr(self, "func_sig")
            func_body = getattr(self, "func_body")
            for dependency in func_deps:
                self.add_codeblock(dependency)
            self.add_single_c_api()
            self.append_code(func_sig)
            self.append_code(func_body)
            return self.clean_code, self.func_deps, self.func_sig, self.func_body
        self.func_deps = self.generate_dependency()
        self.add_single_c_api()
        self.func_sig = self.generate_signature()
        self.func_body = self.generate_body()
        self.verify_code()
        return self.clean_code, self.func_deps, self.func_sig, self.func_body

    def convert_to_device(self):
        assert (
            self.kernel_type == "global"
        ), "Only global kernel can be converted to device"
        self.reset(mode="device")
        self.add_single_c_api()
        self.append_code(GlobalKernel.device_decorator)
        self.declare_return_with_launch_bounds()
        self.declare_name_args()
        self.new_codeblock()
        self.add_line_with_indent(f"if (thread_idx >= {self.block_size})")
        self.new_codeblock()
        self.add_line_with_indent("return;", end=True)
        self.close_codeblock()
        self.shadow_global_dims()
        self.alloc_shared_memory()
        self.add_body_without_syncthreads()
        self.close_codeblock()
        return self.clean_code

    def get_function(self):
        code, _func_deps, _func_sig, _func_body = self.get_code()
        compiled_function = CUDACompiler.create_raw(code)
        return compiled_function
