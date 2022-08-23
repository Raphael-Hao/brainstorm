# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Dict, List, Union

from brt.runtime import log

from brt.jit.codegen.cuda import GlobalKernel
from brt.jit.codegen.module import ModuleKernel
from brt.jit.codegen.utils import make_fused_identifier, make_identifier

logger = log.get_logger(__file__)

__all__ = ["HorizFusedKernel"]


class HorizFusedKernel(GlobalKernel):
    def __init__(
        self,
        candidates: List[ModuleKernel],
    ):
        if not hasattr(self, "kernel_type"):
            setattr(self, "kernel_type", "horiz_fuse")
        super().__init__()
        self.candidates = candidates
        will_initialize = self.check_candidates()
        if will_initialize:
            self.initialize()

    def initialize(self):
        self.mangle_candiates()
        self.generate_new_args()
        self.infer_shared_memory()
        self.calcu_launch_bounds()
        self.calcu_culaunch_dims()
        self.initialized = True

    def mangle_candiates(self):
        for i, module_kernel in enumerate(self.candidates):
            module_kernel.perform_mangling(i)

    def check_candidates(self):
        self.input_infos = []
        self.output_infos = []
        self.parammeters = []
        self.platform = self.candidates[0].platform
        will_initialize = True
        for _, module_func in enumerate(self.candidates):
            if not module_func.initialized:
                will_initialize = False
            assert (
                module_func.platform == self.platform
            ), "platform not same, only support same platform for fuse"
            self.input_infos.append(module_func.input_infos)
            self.output_infos.append(module_func.output_infos)
            self.parammeters.append(module_func.parameters)
        self.module_name = (
            "_".join(module.module_name for module in self.candidates)
            + self.kernel_type
        )
        self.func_name = (
            "_".join(func.func_name for func in self.candidates) + self.kernel_type
        )
        return will_initialize

    def generate_new_args(self):
        self.args = ""
        self.device_args = []
        first_arg_in_global = True
        for i, func in enumerate(self.candidates):
            device_arg = ""
            first_arg_in_device = True
            for _, (arg_type, arg_decorator, arg_name) in enumerate(
                zip(func.arg_types, func.arg_decorators, func.arg_names)
            ):
                if first_arg_in_global:
                    first_arg_in_global = False
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
        for func in self.candidates:
            self.shm_size_in_bytes = max(self.shm_size_in_bytes, func.shm_size_in_bytes)
        self.shm_types = ["char"]
        self.shm_symbols = ["shared_buffer"]
        self.shm_sizes = [self.shm_size_in_bytes]

    def calcu_launch_bounds(self):
        self.max_threads_per_block = 0
        self.min_blocks_per_sm = 1
        for func in self.candidates:
            self.max_threads_per_block = max(
                self.max_threads_per_block, func.max_threads_per_block
            )
            self.min_blocks_per_sm = max(self.min_blocks_per_sm, func.min_blocks_per_sm)

    def calcu_culaunch_dims(self):
        self.grid_size = 0
        self.block_size = 0
        for func in self.candidates:
            self.grid_size += func.grid_size
            self.block_size = max(self.block_size, func.block_size)
        self.blockidx_x = self.grid_size
        self.blockidx_y = 1
        self.blockidx_z = 1
        self.threadidx_x = self.block_size
        self.threadidx_y = 1
        self.threadidx_z = 1

    def generate_dependency(self, sync_method="asm"):
        dependencies = []
        dependencies.append(self.add_codeblock(GlobalKernel.asm_block_sync))
        dependencies.append(self.add_codeblock(GlobalKernel.asm_warp_sync))
        self.new_line()
        for _, func in enumerate(self.candidates):
            device_code = func.convert_to_device()
            dependencies.append(self.add_codeblock(device_code))
        return dependencies

    def generate_body(self):
        formated_code = self.new_codeblock()
        formated_code += self.set_kernel_type()
        formated_code += self.set_culaunch_dims()
        formated_code += self.alloc_shared_memory()
        block_start = 0
        block_end = 0
        for i, func in enumerate(self.candidates):
            block_end = block_start + func.grid_size - 1
            logger.debug(
                f"Fusing blocks from {block_start} to {block_end} for {i}-th block"
            )
            if block_start == block_end:
                if i == 0:
                    formated_code += self.add_line_with_indent(
                        f"if (blockIdx.x == {block_start})", end=True
                    )
                else:
                    formated_code += self.add_line_with_indent(
                        f"else if (blockIdx.x == {block_start})", end=True
                    )
            else:
                if i == 0:
                    formated_code += self.add_line_with_indent(
                        f"if (blockIdx.x <= {block_end})", end=True
                    )
                else:
                    formated_code += self.add_line_with_indent(
                        f"else if (blockIdx.x >= {block_start} && blockIdx.x <= {block_end})",
                        end=True,
                    )
            formated_code += self.new_codeblock()
            # pass shared memory ptr if needed else nullptr
            device_arg = self.device_args[i]
            if func.shm_size_in_bytes > 0:
                device_arg += f", shared_buffer"
            else:
                device_arg += ", nullptr"
            device_arg += f", blockIdx.x - {block_start}, threadIdx.x"
            formated_code += self.add_line_with_indent(
                f"{func.func_name}({device_arg});", end=True
            )
            formated_code += self.close_codeblock()
            block_start = block_end + 1
        assert (
            block_start == self.grid_size
        ), f"block_fused: {block_start} != grid_size: {self.grid_size}"
        formated_code += self.close_codeblock()
        return formated_code

    def make_identifier(self):
        identifiers = [
            make_identifier(
                module.module_name,
                module.input_infos,
                module.output_infos,
                module.parameters,
            )
            for module in self.candidates
        ]
        return make_fused_identifier(identifiers=identifiers)
