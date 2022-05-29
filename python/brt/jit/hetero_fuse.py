# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

from .horiz_fuse import HorizFuseModuleFunction
from .module_func import ModuleFunction


class HeteroFuseModuleFunction(HorizFuseModuleFunction):
    def __init__(self, candidates: List[ModuleFunction]):
        if not hasattr(self, "kernel_type"):
            setattr(self, "kernel_type", "hetero_fuse")
        super().__init__(candidates=candidates)

    def generate_new_args(self):
        super().generate_new_args()
        for i, func in enumerate(self.candidates):
            self.args += f", uint {func.func_name}_active = 1"

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
    
    def generate_signature(self):
        return super().generate_signature()
    
    def generate_body(self):
        self.new_codeblock()
        self.set_kernel_type()
        self.set_culaunch_dims()
        self.alloc_shared_memory()
        block_start = 0
        for i, func in enumerate(self.candidates):
            if i == 0:
                condition = f"({func.func_name}_active * {self.grid_size[i]})"
                self.clean_code += f"  if (blockIdx.x < {condition}) "
            else:
                block_start = "("
                for j in range(i):
                    if j != 0:
                        block_start += " + "
                    block_start += (
                        f"{self.candidates[j].func_name}_active * {self.grid_size[j]}"
                    )
                condition = (
                    f"{block_start} + {func.func_name}_active * {self.grid_size[i]})"
                )
                block_start += ")"
                self.clean_code += f"  else if (blockIdx.x < {condition}) "
            self.new_codeblock()
            device_arg = self.device_args[i]
            if func.shm_size_in_bytes > 0:
                device_arg += f", shared_buffer"
            else:
                device_arg += ", nullptr"
            device_arg += f", blockIdx.x - {block_start}, threadIdx.x"
            self.clean_code += f"    {func.func_name}({device_arg});\n"
            self.close_codeblock()
        self.close_codeblock()

    def make_identifier(self):
        return super().make_identifier()
    