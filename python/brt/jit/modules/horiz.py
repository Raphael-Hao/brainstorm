import itertools
from typing import List, Tuple, Union, Literal, Callable, Any

import torch
from torch import nn
from torch import autograd

from brt.jit.codegen.horiz_fused import HorizFusedKernel
from brt.jit.modules.base import FuseModuleInputType
from brt.jit.modules.fused import FusedModule


class HorizFusedModule(FusedModule):
    def _make_global_kernel(
        self,
        sample_inputs: FuseModuleInputType,
        method: str = "forward",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> HorizFusedKernel:
        assert self.num_submodule == len(
            sample_inputs
        ), "modules and sample_inputs must have the same length"
        if isinstance(rank, int):
            rank = [rank] * self.num_submodule
        candidates = []
        for jsm, inp, rk in zip(self.jit_submodules, sample_inputs, rank):
            module_kernel = jsm._make_global_kernel(inp, method, objective_func, rk)
            candidates.append(module_kernel)
        fused_kernel = HorizFusedKernel(candidates)
        return fused_kernel

    def make_function(
        self,
        sample_inputs: FuseModuleInputType,
        mode: Literal["eval", "train"] = "eval",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> autograd.Function:
        jit_kernel = self.make_kernel(
            sample_inputs=sample_inputs,
            method="forward",
            objective_func=objective_func,
            rank=rank,
        )
        (
            input_arg_num,
            total_arg_num,
            input_arg_indices,
            output_arg_indices,
        ) = self._extract_arg_infos("forward")
        out_data = [
            torch.empty(shp).to("cuda")
            for shp in self._get_output_shape("forward", sample_inputs)
        ]

        class JitFunction(autograd.Function):
            @staticmethod
            def forward(ctx: Any, *inputs):
                inputs = list(inputs)
                for i, out_index in enumerate(output_arg_indices):
                    inputs.insert(out_index, out_data[i])
                jit_kernel(*inputs)
                outputs = [inputs[i] for i in output_arg_indices]
                return tuple(outputs)

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:
                raise NotImplementedError

        return JitFunction

    def make_module(
        self,
        sample_inputs: FuseModuleInputType,
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> nn.Module:
        jit_function = self.make_function(
            sample_inputs=sample_inputs,
            mode="eval",
            objective_func=objective_func,
            rank=rank,
        )
        module_name = f"BRT.HorizFused_{self.num_submodule}"
        extra_repr = "\n".join([str(m) for m in self.module])

        (
            input_arg_num,
            total_arg_num,
            input_arg_indices,
            output_arg_indices,
        ) = self._extract_arg_infos("forward")

        module_input_arg_indices = [
            in_index
            - sum(1 for out_index in output_arg_indices if out_index < in_index)
            for in_index in input_arg_indices
        ]

        class JitHorizFusedModule(nn.Module):
            def __init__(self, *params: torch.Tensor) -> None:
                super().__init__()
                self._extra_repr = extra_repr
                self.function = jit_function
                for i, tensor in enumerate(params):
                    self.register_parameter(f"param_{i}", tensor)
                self._function_input_templete = {}
                self._function_input_templete["forward"] = list(self.parameters())
                for input_index in module_input_arg_indices:
                    self._function_input_templete["forward"].insert(input_index, None)
                # self.forward(list(itertools.chain.from_iterable(sample_inputs)))

            def forward(self, *inputs: torch.Tensor):
                for input, input_index in zip(inputs, module_input_arg_indices):
                    self._function_input_templete["forward"][input_index] = input
                output = self.function.apply(*self._function_input_templete["forward"])
                for input_index in module_input_arg_indices:
                    self._function_input_templete["forward"][input_index] = None
                return output

            def _get_name(self):
                return module_name

            def extra_repr(self):
                return self._extra_repr

        submodule_params_and_buffs = list(
            itertools.chain.from_iterable(
                jsm._extract_parameters_and_buffers() for jsm in self.jit_submodules
            )
        )
        return JitHorizFusedModule(*submodule_params_and_buffs)

    def _extract_shared_arg_infos(
        self,
        method: str,
        sample_inputs: FuseModuleInputType,
    ) -> Tuple[List, List]:
        raise NotImplementedError()

    def _extract_arg_infos(
        self,
        method: str,
    ) -> Tuple[int, int, List[int], List[int]]:
        input_arg_num = 0
        total_arg_num = 0
        input_arg_indices = []
        output_arg_indices = []
        for jsm in self.jit_submodules:
            (
                sub_input_arg_num,
                sub_total_arg_num,
                sub_input_arg_indices,
                sub_output_arg_indices,
            ) = jsm._extract_arg_infos(method)
            output_arg_indices.extend(
                [i + total_arg_num for i in sub_output_arg_indices]
            )
            input_arg_indices.extend([i + total_arg_num for i in sub_input_arg_indices])
            total_arg_num += sub_total_arg_num
            input_arg_num += sub_input_arg_num
        return (
            input_arg_num,
            total_arg_num,
            input_arg_indices,
            output_arg_indices,
        )

    def _get_output_shape(
        self, method: str, sample_inputs: FuseModuleInputType
    ) -> List[torch.Size]:
        return list(
            itertools.chain.from_iterable(
                jsm._get_output_shape(method, sample_input)
                for jsm, sample_input in zip(self.jit_submodules, sample_inputs)
            )
        )

    @property
    def module_name(self) -> str:
        return f"HorizFused_{self.num_submodule}_" + "_".join(
            [jsm.module_name for jsm in self.jit_submodules]
        )
