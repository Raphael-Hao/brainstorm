# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union, Literal, Optional

import torch
from torch import nn

from brt.runtime import log
from brt.jit.modules.atom import AtomModule
from brt.jit.codegen.module import ModuleKernel, ModuleDTypeSizeInByte

logger = log.get_logger(__file__)


class Conv2dElementWiseModule(AtomModule):

    _input_arg_indices = {"forward": [0]}
    _output_arg_indices = {"forward": [2]}
    _shared_arg_indices = {"forward": [0, 2]}

    _required_module_cls = torch.nn.Conv2d
    _optional_succeed_module_cls = (
        torch.nn.BatchNorm2d,
        torch.nn.ReLU,
        torch.nn.Sigmoid,
    )

    def __init__(self, module: nn.Module):
        super().__init__(module)
        if not isinstance(self.module, torch.nn.Sequential):
            self.module = torch.nn.Sequential(self.module)

    @classmethod
    def ismodule(cls, module: torch.nn.Module):
        if not isinstance(module, torch.nn.Sequential):
            module = torch.nn.Sequential(module)
        for i, sub_module in enumerate(module):
            if i == 0:
                if not isinstance(sub_module, cls._required_module_cls):
                    return False
            elif not isinstance(sub_module, cls._optional_succeed_module_cls):
                return False
        return True

    def _make_global_kernel(
        self,
        sample_inputs: torch.Tensor,
        method: str,
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> ModuleKernel:
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        if not isinstance(self.module, torch.nn.Sequential):
            self.module = torch.nn.Sequential(self.module)
        conv2d = self.module[0]
        parameters = {}
        parameters["in_channels"] = conv2d.in_channels
        parameters["out_channels"] = conv2d.out_channels
        parameters["kernel_size"] = conv2d.kernel_size
        parameters["stride"] = conv2d.stride
        parameters["padding"] = conv2d.padding
        parameters["dilation"] = conv2d.dilation
        parameters["groups"] = conv2d.groups

        sample_output = self.module(sample_inputs)
        input_infos = {"input_0": list(sample_inputs.shape)}
        output_infos = {"output_0": list(sample_output.shape)}
        logger.debug(
            f"""
module name: {self.module_name}
input_infos: {input_infos}
output_infos: {output_infos}
parameters: {parameters}
"""
        )
        return ModuleKernel(
            module_name=self.module_name,
            method=method,
            kernel_source=None,
            input_infos=input_infos,
            output_infos=output_infos,
            parameters=parameters,
        ).load_from_db(objective_func, rank)

    def make_module(
        self,
        sample_inputs: torch.Tensor,
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> nn.Module:
        jit_function = self.make_function(
            sample_inputs=sample_inputs,
            mode="eval",
            objective_func=objective_func,
            rank=rank,
        )
        module_name = "BRT." + self.module_name
        extra_repr = self.module[0].extra_repr()

        class JitConv2dElemwiseModule(nn.Module):
            def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None) -> None:
                super().__init__()
                self._extra_repr = extra_repr
                self.function = jit_function
                self.register_parameter("weight", weight)
                self.register_parameter("bias", bias)
                self.forward(sample_inputs)

            def forward(self, input: torch.Tensor):
                if self.bias is not None:
                    return self.function.apply(input, self.weight, self.bias)[0]
                else:
                    return self.function.apply(input, self.weight)[0]

            def _get_name(self):
                return module_name

            def extra_repr(self):
                return self._extra_repr

        return JitConv2dElemwiseModule(*self._extract_parameters_and_buffers())

    def _extract_shared_arg_infos(self, method: str, sample_input: torch.Tensor):
        # Only support method == 'forward' actually
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        sample_output = self.module(sample_input)
        sample_input_size = sample_input.numel() / sample_input.shape[1]
        sample_output_size = sample_output.numel() / sample_output.shape[1]
        shared_arg_grans = [
            sample_input_size * ModuleDTypeSizeInByte[sample_input.dtype],
            sample_output_size * ModuleDTypeSizeInByte[sample_output.dtype],
        ]

        return type(self)._shared_arg_indices[method], shared_arg_grans

    def _extract_arg_infos(self, method: str) -> Tuple[int, int, List[int], List[int]]:
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        if "Bias" not in self.module_name:
            input_arg_num = 2
        else:
            input_arg_num = 3
        total_arg_num = input_arg_num + 1
        return (
            input_arg_num,
            total_arg_num,
            type(self)._input_arg_indices[method],
            type(self)._output_arg_indices[method],
        )

    def _extract_parameters_and_buffers(self) -> List[Optional[torch.Tensor]]:
        ret = [self.module[0].weight]
        if self.module[0].bias is not None:
            ret.append(self.module[0].bias)
        return ret

    @property
    def module_name(self) -> str:
        for i, sub_module in enumerate(self.module):
            if i == 0:
                module_name = "Conv2d"
                if sub_module.bias is not None:
                    module_name += "Bias"
            elif isinstance(sub_module, type(self)._optional_succeed_module_cls):
                if isinstance(sub_module, torch.nn.BatchNorm2d):
                    module_name += "BatchNorm"
                else:
                    module_name += type(sub_module).__name__
        return module_name
