# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Union

from brt.common import log

from .horiz_fused import HorizFusedModuleFunction
from .module_func import ModuleFunction
from .utils import check_if_pointer

logger = log.get_logger(__file__)


class HomoFusedModuleFunction(HorizFusedModuleFunction):
    def __init__(
        self,
        module_name,
        dst_num,  # branch num, e.g., expert num, for horizontal fusion
        capacities: List[int],  # supported dynamic shape
        shared_arg_indices: List[int],
        shared_arg_grans: List[int],
        input_infos: Dict[str, List[int]] = None,
        output_infos: Dict[str, List[int]] = None,
        parameters: Dict[str, List[Union[int, float]]] = None,
    ):
        if not hasattr(self, "kernel_type"):
            setattr(self, "kernel_type", "homo_fuse")
        self.dst_num = dst_num
        self.capacities = capacities
        self.supported_capacity_num = len(self.capacities)
        self.shared_arg_indices = shared_arg_indices
        self.shared_arg_grans = shared_arg_grans
        candidates = self.generate_candidates(
            module_name,
            capacities,
            input_infos,
            output_infos,
            parameters,
        )
        super().__init__(candidates=candidates)

    @staticmethod
    def generate_candidates(
        module_base_name,
        candidates_capacities,
        input_infos: Dict[str, List[int]] = None,
        output_infos: Dict[str, List[int]] = None,
        parameters: Dict[str, List[Union[int, float]]] = None,
    ):
        candidates: List[ModuleFunction] = []
        for dim in candidates_capacities:
            for input_name in input_infos.keys():
                input_infos[input_name] = [dim] + input_infos[input_name][1:]
            for output_name in output_infos.keys():
                output_infos[output_name] = [dim] + output_infos[output_name][1:]
            candidate = ModuleFunction(
                module_base_name,
                input_infos=input_infos,
                output_infos=output_infos,
                parameters=parameters,
            )
            candidate.load_from_db()
            assert candidate.initialized is True, "candidate not initialized"
            candidates.append(candidate)
        return candidates

    def check_candidates(self):
        will_initialize = super().check_candidates()
        self.module_name = (
            self.candidates[0].module_name
            + "_"
            + str(self.supported_capacity_num)
            + "_"
            + str(self.dst_num)
            + "_"
            + self.kernel_type
        )
        self.set_base_args()
        self.verify_homo_args()
        return will_initialize

    def set_base_args(self):
        self.base_arg_types = self.candidates[0].arg_types
        self.base_arg_decorators = self.candidates[0].arg_decorators
        self.base_arg_names = self.candidates[0].arg_names
        self.base_arg_num = len(self.base_arg_types)

    def verify_homo_args(self):
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
                f", uint {self.func_name}_{self.capacities[capacity_idx]}_active_blocks"
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

    def set_branch_config(
        self,
    ):
        formated_code = self.add_line_with_indent(
            f"// [homo_fuse_info] branch_num = {self.dst_num}", end=True
        )
        formated_code += self.add_line_with_indent(
            f"// [homo_fuse_info] supported_capacity = {self.capacities}", end=True
        )
        formated_code += self.add_line_with_indent(
            f"// [homo_fuse_info] arg_num = {self.base_arg_num}", end=True
        )
        formated_code += self.add_line_with_indent(
            f"// [homo_fuse_info] shared_arg_num = {len(self.shared_arg_indices)}",
            end=True,
        )
        formated_code += self.add_line_with_indent(
            f"// [homo_fuse_info] shared_arg_grans = {self.shared_arg_grans}", end=True
        )
        return formated_code

    def generate_signature(self):
        return super().generate_signature()

    def generate_body(self):
        formated_code = self.new_codeblock()
        formated_code += self.set_kernel_type()
        formated_code += self.set_branch_config()
        formated_code += self.set_culaunch_dims()
        formated_code += self.alloc_shared_memory()
        block_start = 0
        for capacity_idx, func in enumerate(self.candidates):
            if capacity_idx == 0:
                condition = f"({self.grid_size[capacity_idx]} * {self.func_name}_{self.capacities[capacity_idx]}_active_blocks)"
                formated_code += self.add_line_with_indent(
                    f"if (blockIdx.x < {condition}) "
                )
            else:
                block_start = ""
                for j in range(capacity_idx):
                    if j != 0:
                        block_start += " + "
                    block_start += f"{self.grid_size[j]} * {self.func_name}_{self.capacities[j]}_active_blocks"
                condition = f"({block_start} + {self.grid_size[capacity_idx]} * {self.func_name}_{self.capacities[capacity_idx]}_active_blocks)"
                formated_code += self.add_line_with_indent(
                    f"else if (blockIdx.x < {condition}) "
                )
            formated_code += self.new_codeblock()
            formated_code += self.add_line_with_indent(
                f"auto blockidxx = blockIdx.x - ({block_start});", end=True
            )
            formated_code += self.add_line_with_indent(
                f"auto arg_idx = blockidxx / {self.grid_size[capacity_idx]}"
            )
            for j in range(capacity_idx):
                formated_code += self.append_code(
                    f" + {self.func_name}_{self.capacities[j]}_active_blocks"
                )
            formated_code += self.append_code(";", end=True)
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
            formated_code += self.add_line_with_indent(
                f"{func.func_name}({device_arg});", end=True
            )
            formated_code += self.close_codeblock()
        formated_code += self.close_codeblock()
        return formated_code

    def make_identifier(self):
        return super().make_identifier()
