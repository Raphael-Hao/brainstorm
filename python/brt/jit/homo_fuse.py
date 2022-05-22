# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

from brt.common import log

from .base import GlobalFunction
from .compiler import CUDACompiler
from .horiz_fuse import HorizFuseFunction
from .raw_func import RawFunction
from .template import Templator
from .utils import check_if_pointer

logger = log.get_logger(__file__)

# TODO should be used for hetro with dynamic shape
class HomoFuseFunctionV1(HorizFuseFunction):
    def __init__(
        self,
        homo_func_name,
        branch_num,
        capacities: List[int],
        shared_arg_indices: List[int],
    ):
        self.name = homo_func_name
        self.branch_num = branch_num
        self.capacities = capacities
        candidates = self.generate_candidates(homo_func_name, capacities)
        super().__init__(candidates=candidates)
        self.shared_arg_indices = shared_arg_indices
        self.set_base_args()
        self.verify_candidates()

    def generate_candidates(self, candidate_base_name, candidates_dims):
        candidates: List[GlobalFunction] = []
        for dim in candidates_dims:
            candidate = Templator.get_global_function(candidate_base_name + f"_{dim}")
            candidates.append(candidate)
        return candidates

    def set_base_args(self):
        self.base_arg_types = self.candidates[0].arg_types
        self.base_arg_decorators = self.candidates[0].arg_decorators
        self.base_arg_names = self.candidates[0].arg_names
        self.base_arg_num = len(self.base_arg_types)

    def verify_candidates(self):
        try:
            assert len(self.shared_arg_indices) > 0
            assert self.base_arg_num > len(self.shared_arg_indices)
            for func in self.candidates:
                assert func.arg_types == self.base_arg_types
                assert func.arg_decorators == self.base_arg_decorators
                assert func.arg_names == self.base_arg_names
            for arg_types in self.base_arg_types:
                assert check_if_pointer(arg_types)
        except AssertionError as e:
            logger.exception(e)

    def generate_new_name(self):
        pass

    def generate_new_args(self):
        self.shared_arg_types = []
        self.shared_arg_decorators = []
        self.shared_arg_names = []
        self.standalone_arg_types = []
        self.standalone_arg_decorators = []
        self.standalone_arg_names = []
        self.device_args = ["" for _ in range(self.branch_num)]
        for arg_index, (arg_type, arg_decorator, arg_name) in enumerate(
            zip(self.base_arg_types, self.base_arg_decorators, self.base_arg_names)
        ):
            if arg_index in self.shared_arg_indices:
                self.shared_arg_types.append(arg_type)
                self.shared_arg_decorators.append(arg_decorator)
                self.shared_arg_names.append(arg_name)
            else:
                self.standalone_arg_types.append(arg_type)
                self.standalone_arg_decorators.append(arg_decorator)
                self.standalone_arg_names.append(arg_name)
            for i in range(self.branch_num):
                if arg_index != 0:
                    self.device_args[i] += ", "
                self.device_args[i] += f"{arg_name}[{i}]"
        self.args = ""
        self.arg_types = self.shared_arg_types + self.standalone_arg_types
        self.arg_decorators = (
            self.shared_arg_decorators + self.standalone_arg_decorators
        )
        self.arg_names = self.shared_arg_names + self.standalone_arg_names
        for arg_index, (arg_type, arg_decorator, arg_name) in enumerate(
            zip(self.arg_types, self.arg_decorators, self.arg_names)
        ):
            if arg_index != 0:
                self.args += ", "
            self.args += f"{arg_type} {arg_decorator} {arg_name}[]"
        for i in range(self.branch_num):
            self.args += f", uint {self.name}_{i}_capacity"

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

    def alloc_array_for_capacity_dims(self):
        self.add_line_with_indent(
            f"int capacity_dims[{len(self.capacities)}]", end=True
        )
        for i in range(len(self.capacities)):
            self.add_line_with_indent(
                f"capacity_dims[{i}] = {self.grid_size[i]}", end=True
            )

    def get_code(self, sync_method="asm"):
        self.fuse()
        self.reset_mode("global")
        self.add_codeblock(GlobalFunction.common_defines)
        self.add_c_api()
        self.add_codeblock(GlobalFunction.asm_block_sync)
        self.add_codeblock(GlobalFunction.asm_warp_sync)
        for idx, func in enumerate(self.candidates):
            self.add_codeblock(
                func.get_code(mode="device", device_id=idx, sync_method=sync_method)
            )
        self.add_line_with_indent(GlobalFunction.global_decorator)
        self.set_launch_bounds()
        self.declare_name_args()
        self.new_codeblock()
        self.set_kernel_type("homo_fuse_v1")
        self.set_culaunch_dims()
        self.alloc_array_for_capacity_dims()
        self.alloc_shared_memory()
        block_start = 0
        for branch_idx in range(self.branch_num):
            condition = f"(capacity_dims[{self.name}_{branch_idx}_capacity])"
            if branch_idx == 0:
                self.add_line_with_indent(f"if (blockIdx.x < {condition}) ")
            else:
                block_start = "("
                for j in range(branch_idx):
                    if j != 0:
                        block_start += " + "
                    block_start += f"capacity_dims[{self.name}_{j}_capacity]"
                condition = (
                    f"{block_start} + capacity_dims[{self.name}_{branch_idx}_capacity])"
                )
                block_start += ")"
                self.add_line_with_indent(f"else if (blockIdx.x < {condition}) ")
            self.new_codeblock()
            self.add_line_with_indent(f"switch ({self.name}_{branch_idx}_capacity) ")
            self.new_codeblock()
            device_arg = self.device_args[branch_idx]
            if func.shm_size_in_bytes > 0:
                device_arg += f", shared_buffer"
            else:
                device_arg += ", nullptr"
            device_arg += f", blockIdx.x - {block_start}, threadIdx.x"
            for capacity_idx, func in enumerate(self.candidates):
                self.add_line_with_indent(f"case {capacity_idx}: ")
                self.new_codeblock()
                self.add_line_with_indent(f"{func.name}({device_arg})", end=True)
                self.close_codeblock()
            self.close_codeblock()
            self.close_codeblock()
        self.close_codeblock()
        self.end_c_api()
        self.verify_code()
        return self.clean_code


class HomoFuseFunctionV2(HorizFuseFunction):
    def __init__(
        self,
        homo_func_name,
        branch_num,  # branch num, e.g., expert num, for horizontal fusion
        capacities: List[int],  # supported dynamic shape
        shared_arg_indices: List[int],
        shared_arg_grans: List[int],
    ):
        self.name = homo_func_name
        self.branch_num = branch_num
        self.capacities = capacities
        self.supported_capacity_num = len(self.capacities)
        self.shared_arg_indices = shared_arg_indices
        self.shared_arg_grans = shared_arg_grans
        candidates = self.generate_candidates(homo_func_name, capacities)
        super().__init__(candidates=candidates)
        self.set_base_args()
        self.verify_candidates()

    def generate_candidates(self, candidate_base_name, candidates_dims):
        candidates: List[GlobalFunction] = []
        for dim in candidates_dims:
            candidate = Templator.get_global_function(candidate_base_name + f"_{dim}")
            candidates.append(candidate)
        return candidates

    def set_base_args(self):
        self.base_arg_types = self.candidates[0].arg_types
        self.base_arg_decorators = self.candidates[0].arg_decorators
        self.base_arg_names = self.candidates[0].arg_names
        self.base_arg_num = len(self.base_arg_types)

    def verify_candidates(self):
        try:
            assert len(self.shared_arg_indices) > 0
            assert self.base_arg_num > len(self.shared_arg_indices)
            for func in self.candidates:
                assert func.arg_types == self.base_arg_types
                assert func.arg_decorators == self.base_arg_decorators
                assert func.arg_names == self.base_arg_names
            for arg_types in self.base_arg_types:
                assert check_if_pointer(arg_types)
        except AssertionError as e:
            logger.exception(e)

    def generate_new_name(self):
        pass

    def generate_new_args(self):
        self.shared_arg_types = []
        self.shared_arg_decorators = []
        self.shared_arg_names = []
        self.standalone_arg_types = []
        self.standalone_arg_decorators = []
        self.standalone_arg_names = []
        self.device_args = []
        for arg_index, (arg_type, arg_decorator, arg_name) in enumerate(
            zip(self.base_arg_types, self.base_arg_decorators, self.base_arg_names)
        ):
            if arg_index in self.shared_arg_indices:
                self.shared_arg_types.append(arg_type)
                self.shared_arg_decorators.append(arg_decorator)
                self.shared_arg_names.append(arg_name)
            else:
                self.standalone_arg_types.append(arg_type)
                self.standalone_arg_decorators.append(arg_decorator)
                self.standalone_arg_names.append(arg_name)
            self.device_args.append(arg_name)
        self.args = ""
        self.arg_types = self.shared_arg_types + self.standalone_arg_types
        self.arg_decorators = (
            self.shared_arg_decorators + self.standalone_arg_decorators
        )
        self.arg_names = self.shared_arg_names + self.standalone_arg_names
        for arg_index, (arg_type, arg_decorator, arg_name) in enumerate(
            zip(self.arg_types, self.arg_decorators, self.arg_names)
        ):
            if arg_index != 0:
                self.args += ", "
            self.args += f"{arg_type} {arg_decorator} {arg_name}[]"
        # static array , length equal to the number of capacity
        for capacity_idx in range(self.supported_capacity_num):
            self.args += (
                f", uint {self.name}_{self.capacities[capacity_idx]}_active_blocks"
            )

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

    def set_branch_num(self,):
        self.add_line_with_indent(
            f"// [homo_fuse_info] branch_num = {self.branch_num}", end=True
        )
        self.add_line_with_indent(
            f"// [homo_fuse_info] supported_capacity = {self.capacities}", end=True
        )
        self.add_line_with_indent(
            f"// [homo_fuse_info] arg_num = {self.base_arg_num}", end=True
        )
        self.add_line_with_indent(
            f"// [homo_fuse_info] shared_arg_num = {len(self.shared_arg_indices)}",
            end=True,
        )
        self.add_line_with_indent(
            f"// [homo_fuse_info] shared_arg_grans = {self.shared_arg_grans}", end=True
        )

    def get_code(self, sync_method="asm"):
        self.fuse()
        self.reset_mode("global")
        self.add_codeblock(GlobalFunction.common_defines)
        self.add_c_api()
        self.add_codeblock(GlobalFunction.asm_block_sync)
        self.add_codeblock(GlobalFunction.asm_warp_sync)
        for idx, func in enumerate(self.candidates):
            self.add_codeblock(
                func.get_code(mode="device", device_id=idx, sync_method=sync_method)
            )
        self.add_line_with_indent(GlobalFunction.global_decorator)
        self.set_launch_bounds()
        self.declare_name_args()
        self.new_codeblock()
        self.set_kernel_type("homo_fuse_v2")
        self.set_branch_num()
        self.set_culaunch_dims()
        self.alloc_shared_memory()
        block_start = 0
        for capacity_idx, func in enumerate(self.candidates):
            if capacity_idx == 0:
                condition = f"({self.grid_size[capacity_idx]} * {self.name}_{self.capacities[capacity_idx]}_active_blocks)"
                self.add_line_with_indent(f"if (blockIdx.x < {condition}) ")
            else:
                block_start = ""
                for j in range(capacity_idx):
                    if j != 0:
                        block_start += " + "
                    block_start += f"{self.grid_size[j]} * {self.name}_{self.capacities[j]}_active_blocks"
                condition = f"({block_start} + {self.grid_size[capacity_idx]} * {self.name}_{self.capacities[capacity_idx]}_active_blocks)"
                self.add_line_with_indent(f"else if (blockIdx.x < {condition}) ")
            self.new_codeblock()
            self.add_line_with_indent(
                f"auto blockidxx = blockIdx.x - ({block_start});", end=True
            )
            self.add_line_with_indent(
                f"auto arg_idx = blockidxx / {self.grid_size[capacity_idx]}"
            )
            for j in range(capacity_idx):
                self.add_code(f" + {self.name}_{self.capacities[j]}_active_blocks")
            self.add_code(";", end=True)
            device_arg = ""
            for arg_i, arg in enumerate(self.device_args):
                if arg_i != 0:
                    device_arg += ", "
                device_arg += f"{arg}[arg_idx]"
            if func.shm_size_in_bytes > 0:
                device_arg += f", shared_buffer"
            else:
                device_arg += ", nullptr"
            device_arg += f", blockidxx, threadIdx.x"
            self.add_line_with_indent(f"{func.name}({device_arg});", end=True)
            self.close_codeblock()
        self.close_codeblock()
        self.end_c_api()
        self.verify_code()
        return self.clean_code


class ElaticHomoFuseFunction(HorizFuseFunction):
    def __init__(self, fuse_cadidate_templates: List[str]):
        pass
