# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

from .base import GlobalFunction
from .horiz_fuse import HorizFuseModuleFunction


class HeteroFuseModuleFunction(HorizFuseModuleFunction):
    def __init__(self, candidates: List[Union[GlobalFunction, str]]):
        super().__init__(candidates=candidates)

    def generate_new_args(self):
        super().generate_new_args()
        for i, func in enumerate(self.candidates):
            self.args += f", uint {func.name}_active = 1"

    def calcu_culaunch_dims(self):
        self.grid_size = []
        self.block_size = []
        for func in self.candidates:
            self.grid_size.append(func.grid_size)
            self.block_size.append(func.block_size)
        self.blockidx_x = self.grid_size
        self.blockidx_y = 1
        self.blockidx_z = 1
        self.threadidx_x = self.block_size
        self.threadidx_y = 1
        self.threadidx_z = 1

    def get_code(self, sync_method="asm"):
        self.fuse()
        self.reset_mode("global")
        self.clean_code += GlobalFunction.common_defines
        self.clean_code += GlobalFunction.c_api_decorator
        self.new_codeblock()
        self.clean_code += GlobalFunction.asm_block_sync
        self.clean_code += GlobalFunction.asm_warp_sync
        self.new_emtpy_line()
        for idx, func in enumerate(self.candidates):
            self.clean_code += func.get_code(
                mode="device", device_id=idx, sync_method=sync_method
            )
        self.clean_code += GlobalFunction.global_decorator
        self.set_launch_bounds()
        self.declare_name_args()
        self.new_codeblock()
        self.set_kernel_type("hetero_fuse")
        self.set_culaunch_dims()
        self.alloc_shared_memory()
        block_start = 0
        for i, func in enumerate(self.candidates):
            if i == 0:
                condition = f"({func.name}_active * {self.grid_size[i]})"
                self.clean_code += f"  if (blockIdx.x < {condition}) "
            else:
                block_start = "("
                for j in range(i):
                    if j != 0:
                        block_start += " + "
                    block_start += (
                        f"{self.candidates[j].name}_active * {self.grid_size[j]}"
                    )
                condition = f"{block_start} + {func.name}_active * {self.grid_size[i]})"
                block_start += ")"
                self.clean_code += f"  else if (blockIdx.x < {condition}) "
            self.new_codeblock()
            device_arg = self.device_args[i]
            if func.shm_size_in_bytes > 0:
                device_arg += f", shared_buffer"
            else:
                device_arg += ", nullptr"
            device_arg += f", blockIdx.x - {block_start}, threadIdx.x"
            self.clean_code += f"    {func.name}({device_arg});\n"
            self.close_codeblock()
        self.close_codeblock()
        self.close_codeblock()
        self.verify_code()
        return self.clean_code
