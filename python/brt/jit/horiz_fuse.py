# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from brt.common import log

from .base import BaseFunction
from .compiler import CUDACompiler
from .generic import GenericFunction

logger = log.get_logger(__file__)


class HorizFuseFunction(BaseFunction):
    def __init__(self, fuse_cadidate_templates: List[str]):
        self.fuse_cadidates: List[GenericFunction] = []
        for i in range(len(fuse_cadidate_templates)):
            func_template = fuse_cadidate_templates[i]
            generic_func = GenericFunction(func_template)
            self.fuse_cadidates.append(generic_func)
            generic_func.name += f"_block_{i}"

    def fuse(self):
        self.generate_new_name()
        self.generate_new_args()
        self.infer_shared_memory()
        self.calcu_culaunch_dims()

    def generate_new_name(self):
        self.name = ""
        first_block = True
        for func in self.fuse_cadidates:
            if first_block:
                first_block = False
            else:
                self.name += "_"
            self.name = self.name + func.name

    def generate_new_args(self):
        self.args = ""
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
                    self.args += ", "
                self.args += f"{arg_type} {arg_decorator} {arg_name}_{i}"
                if first_arg_in_device:
                    first_arg_in_device = False
                else:
                    device_arg += ", "
                device_arg += f"{arg_name}_{i}"
            self.device_args.append(device_arg)

    def infer_shared_memory(self):
        self.shm_size_in_bytes = 0
        for func in self.fuse_cadidates:
            self.shm_size_in_bytes = max(self.shm_size_in_bytes, func.shm_size_in_bytes)
        self.shm_types = ["char"]
        self.shm_symbols = ["shared_buffer"]
        self.shm_sizes = [self.shm_size_in_bytes]

    def calcu_culaunch_dims(self):
        self.max_threads_per_block = 0
        self.min_blocks_per_sm = 1
        self.grid_size = 0
        self.block_size = 0
        for func in self.fuse_cadidates:
            self.max_threads_per_block = max(
                self.max_threads_per_block, func.max_threads_per_block
            )
            self.min_blocks_per_sm = max(self.min_blocks_per_sm, func.min_blocks_per_sm)
            self.grid_size += func.grid_size
            self.block_size = max(self.block_size, func.block_size)
        self.blockidx_xdim = self.grid_size
        self.blockidx_ydim = 1
        self.blockidx_zdim = 1
        self.threadidx_xdim = self.block_size
        self.threadidx_ydim = 1
        self.threadidx_zdim = 1

    def get_code(self, sync_method="asm"):
        self.fuse()
        self.reset_mode("global")
        self.clean_code += GenericFunction.common_defines
        self.clean_code += GenericFunction.c_api_decorator
        self.new_codeblock()
        self.clean_code += GenericFunction.asm_block_sync
        self.clean_code += GenericFunction.asm_warp_sync
        self.new_emtpy_line()
        for idx, func in enumerate(self.fuse_cadidates):
            self.clean_code += func.get_code(
                mode="device", device_id=idx, sync_method=sync_method
            )
        self.clean_code += GenericFunction.global_decorator
        self.set_launch_bounds()
        self.declare_name_args()
        self.new_codeblock()
        self.set_culaunch_dims()
        self.alloc_shared_memory()
        block_start = 0
        block_end = 0
        for i, func in enumerate(self.fuse_cadidates):
            block_end = block_start + func.grid_size - 1
            logger.debug(
                f"Fusing blocks from {block_start} to {block_end} for {i}-th block"
            )
            if block_start == block_end:
                if i == 0:
                    self.clean_code += f"  if (blockIdx.x == {block_start})\n"
                else:
                    self.clean_code += f"  else if (blockIdx.x == {block_start})\n"
            else:
                if i == 0:
                    self.clean_code += f"  if (blockIdx.x <= {block_end})\n"
                else:
                    self.clean_code += f"  else if (blockIdx.x >= {block_start} && blockIdx.x <= {block_end})\n"
            self.new_codeblock()
            device_arg = self.device_args[i]
            if func.shm_size_in_bytes > 0:
                device_arg += f", shared_buffer"
            else:
                device_arg += ", NULL"
            device_arg += f", blockIdx.x - {block_start}, threadIdx.x"
            self.clean_code += f"    {func.name}({device_arg});\n"
            self.close_codeblock()
            block_start = block_end + 1
        assert (
            block_start == self.grid_size
        ), f"block_fused: {block_start} != grid_size: {self.grid_size}"
        self.close_codeblock()
        self.close_codeblock()
        return self.clean_code
