# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from brt.common import log

from .base import GlobalFunction
from .compiler import CUDACompiler
from .horiz_fuse import HorizFuseFunction
from .raw_func import RawFunction
from .template import Templator

logger = log.get_logger(__file__)


class HomoFuseFunction(HorizFuseFunction):
    def __init__(self, homo_func_name: str, candidate_dims: List[int]):
        self.name = homo_func_name
        self.candidate_dims = candidate_dims
        candidates = self.generate_candidates()
        super().__init__(candidates=candidates)

    def generate_candidates(self):
        candidates: List[GlobalFunction] = []
        for dim in self.candidate_dims:
            candidate = Templator.get_global_function(self.name + f"_{dim}")
            candidates.append(candidate)
        return candidates

    def runtime_culaunch_dims(self, grid_per_block: List[int]):
        grid_size = 0
        for block_id, (func, grid) in enumerate(self.candidates, grid_per_block):
            self.grid_size += func.grid_size * grid
        return self.grid_size

    def generate_new_args(self):
        self.func_args = ""
        first_device_arg = ""
        self.device_args = []
        first_arg_in_func = True
        for i in range(len(self.candidates)):
            func = self.candidates[i]
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
        for func in self.candidates:
            if first_block:
                first_block = False
            else:
                self.func_name += "_"
            self.func_name = self.func_name + func.name

    def fuse(self):
        self.generate_new_name()
        self.generate_new_args()
        self.alloc_shared_memory()

    def get_code(self):
        self.fuse()
        clean_code = ""
        clean_code += GlobalFunction.common_defines
        clean_code += GlobalFunction.c_api_decorator
        clean_code += "{ \n"
        for func in self.candidates:
            clean_code += func.get_code(mode="device")
        clean_code += GlobalFunction.global_decorator
        clean_code += f"__launch_bounds__({self.launch_bounds})"
        clean_code += f" {self.func_name}("
        clean_code += f"{self.func_args})"
        clean_code += "{\n"
        clean_code += f"  // [thread_extent] blockIdx.xdim = {0}\n"
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
        for i in range(len(self.candidates)):
            func = self.candidates[i]
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
        clean_code += "}\n"
        return clean_code


class ElaticHomoFuseFunction(HorizFuseFunction):
    def __init__(self, fuse_cadidate_templates: List[str]):
        pass
