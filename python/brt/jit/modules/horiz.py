from typing import List, Tuple, Union, Literal, Callable, Type, Any, Dict

import itertools

import torch
from torch import nn
from torch import autograd
from torch.overrides import (
    handle_torch_function,
    wrap_torch_function,
    has_torch_function,
)

from brt.runtime.log import get_logger

from brt.jit.codegen.horiz_fused import HorizFusedKernel
from brt.jit.modules.base import FuseModuleInputType
from brt.jit.modules.fused import FusedModule, JitFusedModule


# Note: The `sampe_inputs` is a list of each candidates' inputs, but the runtime
# inputs of the module's forward function should be like the chain of the `sampe_inputs`

logger = get_logger(__file__)


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
        input_arg_indices_chain = list(itertools.chain.from_iterable(input_arg_indices))
        output_arg_indices_chain = list(
            itertools.chain.from_iterable(output_arg_indices)
        )
        output_shape = list(self._get_output_shape("forward", sample_inputs))
        print(output_shape)
        empty_output_shape = list(torch.Size([0, *oshp[1:]]) for oshp in output_shape)
        out_data = [
            torch.empty(shp).to("cuda")
            for shp in output_shape
        ]

        class JitFunction(autograd.Function):
            @staticmethod
            def forward(ctx: Any, *inputs):
                inputs = list(inputs)
                # if any(
                #     input.numel() == 0
                #     if isinstance(input, torch.Tensor)
                #     else input is None
                #     for input in inputs
                # ):
                for input in inputs:
                    if (
                        isinstance(input, torch.Tensor) and input.numel() == 0
                    # ) or input is None:
                    ):
                        # logger.debug("empty input found, exits")
                        return [torch.empty(eoshp).cuda() for eoshp in empty_output_shape]
                for i, out_index in enumerate(output_arg_indices_chain):
                    inputs.insert(out_index, out_data[i])
                jit_kernel(*inputs)
                outputs = []
                for branch_input_indices, branch_output_indices in zip(
                    input_arg_indices, output_arg_indices
                ):
                    branch_inputs = [inputs[bii] for bii in branch_input_indices]
                    branch_outputs = [inputs[boi] for boi in branch_output_indices]
                    if has_torch_function(branch_inputs):
                        packed_branch_outputs = handle_torch_function(
                            lambda *x: branch_outputs, branch_inputs, *branch_inputs
                        )
                        outputs.extend(packed_branch_outputs)
                # outputs = [inputs[i] for i in output_arg_indices_chain]
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
            - [
                out_index < in_index
                for out_index in itertools.chain.from_iterable(output_arg_indices)
            ].count(True)
            for in_index in itertools.chain.from_iterable(input_arg_indices)
        ]

        submodule_params_and_buffs = list(
            itertools.chain.from_iterable(
                jsm._extract_parameters_and_buffers() for jsm in self.jit_submodules
            )
        )
        return JitHorizFusedModule(
            module_input_arg_indices,
            function=jit_function,
            module_name=module_name,
            extra_repr=extra_repr,
            parameters={
                f"param_{i}": param
                for i, param in enumerate(submodule_params_and_buffs)
            },
        )

    def _extract_shared_arg_infos(
        self,
        method: str,
        sample_inputs: FuseModuleInputType,
    ) -> Tuple[List, List]:
        raise NotImplementedError()

    def _extract_arg_infos(
        self,
        method: str,
    ) -> Tuple[int, int, List[List[int]], List[List[int]]]:
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
            output_arg_indices.append(
                [i + total_arg_num for i in sub_output_arg_indices]
            )
            input_arg_indices.append([i + total_arg_num for i in sub_input_arg_indices])
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


class JitHorizFusedModule(JitFusedModule):
    def __init__(
        self,
        module_input_arg_indices: List[int],
        function: Type[autograd.Function],
        module_name: str = "BRT.HorizFusedModule",
        extra_repr: str = "",
        parameters: Dict[str, torch.Tensor] = ...,
    ):
        super().__init__(function, module_name, extra_repr, parameters)
        self._factory_cls = JitHorizFusedModule
        self._function_input_templete = {}
        self._function_input_templete["forward"] = list(self.parameters())
        self._module_input_arg_indices = module_input_arg_indices
        for input_index in self._module_input_arg_indices:
            self._function_input_templete["forward"].insert(input_index, None)
        # self.forward(list(itertools.chain.from_iterable(sample_inputs)))

    def forward(self, *inputs: torch.Tensor):

        for input, input_index in zip(inputs, self._module_input_arg_indices):
            self._function_input_templete["forward"][input_index] = input
        output = self.function.apply(*self._function_input_templete["forward"])
        for input_index in self._module_input_arg_indices:
            self._function_input_templete["forward"][input_index] = None
        return output
