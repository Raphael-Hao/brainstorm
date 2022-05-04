# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from brt.common import log

from .compiler import CUDACompiler
from .generic import GenericFunction

logger = log.get_logger(__file__)


class BlockFuser:
    def __init__(self, fuse_cadidate_templates: List[str]):
        self.fuse_cadidates = []
        for i in range(len(fuse_cadidate_templates)):
            func_template = fuse_cadidate_templates[i]
            generic_func = GenericFunction(func_template)
            self.fuse_cadidates.append(generic_func)
            generic_func.name += f"_block_{i}"

    def alloc_shared_memory(self):
        self.shm_size_in_bytes = 0
        for func in self.fuse_cadidates:
            self.shm_size_in_bytes = max(self.shm_size_in_bytes, func.shm_size_in_bytes)

    def calcu_culaunch_dims(self):
        self.launch_bounds = 0
        self.grid_size = 0
        self.block_size = 0
        for func in self.fuse_cadidates:
            self.launch_bounds = max(self.launch_bounds, func.launch_bounds)
            self.grid_size += func.grid_size
            self.block_size = max(self.block_size, func.block_size)

    def generate_new_args(self):
        self.func_args = ""
        self.device_args = []
        first_arg_in_func = True
        for i in range(len(self.fuse_cadidates)):
            func = self.fuse_cadidates[i]
            device_arg = ""
            first_arg_in_device = True
            for _, (arg_type, arg_decorator, arg_name) in enumerate(
                zip(func.arg_types, func.arg_decorators, func.arg_names)
            ):
                if first_arg_in_func:
                    first_arg_in_func = False
                else:
                    self.func_args += ", "
                self.func_args += f"{arg_type} {arg_decorator} {arg_name}_{i}"
                if first_arg_in_device:
                    first_arg_in_device = False
                else:
                    device_arg += ", "
                device_arg += f"{arg_name}_{i}"
            self.device_args.append(device_arg)

    def generate_new_name(self):
        self.func_name = ""
        first_block = True
        for func in self.fuse_cadidates:
            if first_block:
                first_block = False
            else:
                self.func_name += "_"
            self.func_name = self.func_name + func.name

    def fuse(self):
        self.generate_new_name()
        self.generate_new_args()
        self.alloc_shared_memory()
        self.calcu_culaunch_dims()

    def get_code(self):
        self.fuse()
        clean_code = ""
        clean_code += GenericFunction.common_defines
        for func in self.fuse_cadidates:
            clean_code += func.get_code(mode="device")
        clean_code += GenericFunction.global_decorator
        clean_code += f"__launch_bounds__({self.launch_bounds})"
        clean_code += f" {self.func_name}("
        clean_code += f"{self.func_args})"
        clean_code += "{\n"
        clean_code += f"  // [thread_extent] blockIdx.xdim = {self.grid_size}\n"
        clean_code += f"  // [thread_extent] blockIdx.ydim = {1}\n"
        clean_code += f"  // [thread_extent] blockIdx.zdim = {1}\n"
        clean_code += f"  // [thread_extent] threadIdx.xdim = {self.block_size}\n"
        clean_code += f"  // [thread_extent] threadIdx.ydim = {1}\n"
        clean_code += f"  // [thread_extent] threadIdx.zdim = {1}\n"
        clean_code += "\n"
        if self.shm_size_in_bytes > 0:
            clean_code += (
                f"  __shared__ char shared_buffer[{self.shm_size_in_bytes}];\n"
            )
            clean_code += "\n"
        block_start = 0
        block_end = 0
        for i in range(len(self.fuse_cadidates)):
            func = self.fuse_cadidates[i]
            block_end = block_start + func.grid_size - 1

            logger.debug(
                f"Fusing blocks from {block_start} to {block_end} for {i}-th block"
            )
            if block_start == block_end:
                if i == 0:
                    clean_code += f"  if (blockIdx.x == {block_start})\n"
                else:
                    clean_code += f"  else if (blockIdx.x == {block_start})\n"
            else:
                if i == 0:
                    clean_code += f"  if (blockIdx.x <= {block_end})\n"
                else:
                    clean_code += f"  else if (blockIdx.x >= {block_start} && blockIdx.x <= {block_end})\n"
            clean_code += "  {\n"
            device_arg = self.device_args[i]
            if func.shm_size_in_bytes > 0:
                device_arg += f", shared_buffer"
            else:
                device_arg += ", NULL"
            device_arg += f", blockIdx.x - {block_start}, threadIdx.x"
            clean_code += f"    {func.name}({device_arg});\n"
            clean_code += "  }\n"
            block_start = block_end + 1
        assert (
            block_start == self.grid_size
        ), f"block_fused: {block_start} != grid_size: {self.grid_size}"
        clean_code += "}\n"

        return clean_code

