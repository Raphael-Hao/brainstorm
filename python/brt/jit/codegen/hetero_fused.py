# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Union

from brt.jit.codegen.cuda import GlobalKernel
from brt.jit.codegen.horiz_fused import HorizFusedKernel
from brt.jit.codegen.module import ModuleKernel

__all__ = ["HeteroFusedKernel"]


class HeteroFusedKernel(HorizFusedKernel):
    def __init__(self, candidates: List[ModuleKernel]):
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


class DynamicHeteroFusedKernel(HorizFusedKernel):
    def __init__(
        self,
        candidate_module_names: List[str],
        candidate_capacities: Dict[str, List[int]],
        input_infos: Dict[str, List[int]] = None,
        output_infos: Dict[str, List[int]] = None,
        parameters: Dict[str, List[Union[int, float]]] = None,
    ):
        if not hasattr(self, "kernel_type"):
            setattr(self, "kernel_type", "dynamic_hetero_fuse")
        candidates = self.generate_candidates(
            candidate_module_names, candidate_capacities
        )
        super().__init__(candidates=candidates)
        self.set_base_args()
        self.verify_candidates()
        self.initialized = True

    def verify_candidates(self):
        return super().verify_candidates()

    def generate_candidates(
        self,
        candidate_module_names,
        candidates_dims,
        input_infos,
        output_infos,
        parameters,
    ):
        candidates: Dict[str, List[ModuleKernel]] = []
        for module_name in candidate_module_names:
            input_infos = self.input_infos[module_name]
            output_infos = self.output_infos[module_name]
            parameters = self.parameters[module_name]
            for dim in candidates_dims[module_name]:
                for input_name in input_infos.keys():
                    input_infos[input_name] = [dim] + input_infos[input_name][1:]
                for output_name in output_infos.keys():
                    output_infos[output_name] = [dim] + output_infos[output_name][1:]
                candidate = ModuleKernel(
                    module_name,
                    method=self.method,
                    input_infos=input_infos,
                    output_infos=output_infos,
                    parameters=parameters,
                )
                candidate.load_from_db()
                assert candidate.initialized is True, "candidate not initialized"
                self.candidates[module_name].append(candidate)
        return candidates

    def check_candidates(self):
        raise NotImplementedError()

    def set_base_args(self):
        self.base_arg_types = self.candidates[0].arg_types
        self.base_arg_decorators = self.candidates[0].arg_decorators
        self.base_arg_names = self.candidates[0].arg_names
        self.base_arg_num = len(self.base_arg_types)

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
            self.args += f", uint {self.func_name}_{i}_capacity"

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

    def get_code(self):
        self.fuse()
        self.reset_mode("global")
        self.add_codeblock(GlobalKernel.common_defines)
        self.add_single_c_api()
        self.add_codeblock(GlobalKernel.asm_block_sync)
        self.add_codeblock(GlobalKernel.asm_warp_sync)
        for idx, func in enumerate(self.candidates):
            self.add_codeblock(func.convert_to_device())
        self.add_line_with_indent(GlobalKernel.global_decorator)
        self.declare_return_with_launch_bounds()
        self.declare_name_args()
        self.new_codeblock()
        self.set_kernel_type("homo_fuse_v1")
        self.set_culaunch_dims()
        self.alloc_array_for_capacity_dims()
        self.alloc_shared_memory()
        block_start = 0
        for branch_idx in range(self.branch_num):
            condition = f"(capacity_dims[{self.func_name}_{branch_idx}_capacity])"
            if branch_idx == 0:
                self.add_line_with_indent(f"if (blockIdx.x < {condition}) ")
            else:
                block_start = "("
                for j in range(branch_idx):
                    if j != 0:
                        block_start += " + "
                    block_start += f"capacity_dims[{self.func_name}_{j}_capacity]"
                condition = f"{block_start} + capacity_dims[{self.func_name}_{branch_idx}_capacity])"
                block_start += ")"
                self.add_line_with_indent(f"else if (blockIdx.x < {condition}) ")
            self.new_codeblock()
            self.add_line_with_indent(
                f"switch ({self.func_name}_{branch_idx}_capacity) "
            )
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
                self.add_line_with_indent(f"{func.func_name}({device_arg})", end=True)
                self.close_codeblock()
            self.close_codeblock()
            self.close_codeblock()
        self.close_codeblock()
        self.verify_code()
        return self.clean_code
