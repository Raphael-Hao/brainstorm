# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.jit.modules.base import ModuleInfo
from brt.jit.codegen.module import ModuleKernel, ModuleDTypeSizeInByte

logger = log.get_logger(__file__)


class Conv2dPReLUInfo(ModuleInfo):
    """Info for fused torch.nn.Conv2d & torch.nn.PReLU kernel

    Method Args:
        forward:
            [0]: float* __restrict__ placeholder,  Input
            [1]: float* __restrict__ placeholder1, Conv2d.weight
            [2]: float* __restrict__ T_prelu,      Output
            [3]: float* __restrict__ placeholder2, Conv2d.bias
            [4]: float* __restrict__ placeholder3, PReLU.weight
    """

    _input_arg_indices = {"forward": [0]}
    _output_arg_indices = {"forward": [2]}
    _shared_arg_indices = {"forward": [0, 2]}

    _involved_module_cls = torch.nn.Conv2d
    _succeed_module_cls = torch.nn.PReLU

    @classmethod
    def ismodule(cls, module: torch.nn.Module):
        if not isinstance(module, torch.nn.Sequential):
            return False
        for i, sub_module in enumerate(module):
            if i == 0:
                if not isinstance(sub_module, cls._involved_module_cls):
                    return False
            elif not isinstance(sub_module, cls._succeed_module_cls):
                return False
        return False

    def make_kernel(
        self,
        method: str,
        sample_input: torch.Tensor,
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> ModuleKernel:
        assert method in type(self)._shared_arg_indices, f"{method} is not supported"
        conv2d = self.module[0]
        parameters = {}
        parameters["in_channels"] = conv2d.in_channels
        parameters["out_channels"] = conv2d.out_channels
        parameters["kernel_size"] = conv2d.kernel_size
        parameters["stride"] = conv2d.stride
        parameters["padding"] = conv2d.padding
        parameters["dilation"] = conv2d.dilation
        parameters["groups"] = conv2d.groups
        # TODO: full support of PReLU
        # parameters['num_parameters'] = module[1].num_parameters

        sample_output = self.module(sample_input)
        input_infos = {"input_0": list(sample_input.shape)}
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

    def extract_shared_arg_infos(self, method: str, sample_input: torch.Tensor):
        assert method in type(self)._shared_arg_indices, f"{method} is not supported"
        sample_output = self.module(sample_input)
        sample_input_size = sample_input.numel() / sample_input.shape[1]
        sample_output_size = sample_output.numel() / sample_output.shape[1]
        shared_arg_grans = [
            sample_input_size * ModuleDTypeSizeInByte[sample_input.dtype],
            sample_output_size * ModuleDTypeSizeInByte[sample_output.dtype],
        ]

        return type(self)._shared_arg_indices[method], shared_arg_grans

    def extract_arg_infos(self, module: torch.nn.Module, method: str):
        assert method in type(self)._shared_arg_indices, f"{method} is not supported"
        input_arg_num = 2
        if "Bias" in self.module_name:
            input_arg_num += 1
        if "PReLU" in self.module_name:
            input_arg_num += 1
        total_arg_num = input_arg_num + 1

        return (
            input_arg_num,
            total_arg_num,
            type(self)._input_arg_indices[method],
            type(self)._output_arg_indices[method],
        )

    def _get_module_name(self) -> str:
        module_name = "Conv2d"
        if self.module[0].bias is not None:
            module_name += "Bias"
        if len(self.module) == 2:
            module_name += "PReLU"
        return module_name
