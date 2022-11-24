# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.jit.modules.base import ModuleInfo
from brt.jit.codegen.module import ModuleKernel, ModuleDTypeSizeInByte

logger = log.get_logger(__file__)


class Conv2dElementWiseInfo(ModuleInfo):
    _required_module_cls = torch.nn.Conv2d
    _optional_succeed_module_cls = (
        torch.nn.BatchNorm2d,
        torch.nn.ReLU,
        torch.nn.Sigmoid,
    )
    _input_arg_indices = {"forward": [0]}
    _output_arg_indices = {"forward": [2]}
    _shared_arg_indices = {"forward": [0, 2]}

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

    @classmethod
    def make_kernel(
        cls,
        module: torch.nn.Module,
        method: str,
        sample_input: torch.Tensor,
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> ModuleKernel:
        assert method in cls._shared_arg_indices, f"{method} is not supported"
        if not isinstance(module, torch.nn.Sequential):
            module = torch.nn.Sequential(module)
        module_name = cls.get_module_name(module)
        parameters = {}
        parameters["in_channels"] = module[0].in_channels
        parameters["out_channels"] = module[0].out_channels
        parameters["kernel_size"] = module[0].kernel_size
        parameters["stride"] = module[0].stride
        parameters["padding"] = module[0].padding
        parameters["dilation"] = module[0].dilation
        parameters["groups"] = module[0].groups

        sample_output = module(sample_input)
        input_infos = {"input_0": list(sample_input.shape)}
        output_infos = {"output_0": list(sample_output.shape)}
        logger.debug(
            f"""
module name: {module_name}
input_infos: {input_infos}
output_infos: {output_infos}
parameters: {parameters}
"""
        )
        return ModuleKernel(
            module_name=module_name,
            method=method,
            kernel_source=None,
            input_infos=input_infos,
            output_infos=output_infos,
            parameters=parameters,
        ).load_from_db(objective_func, rank)

    @classmethod
    def extract_shared_arg_infos(
        cls, module: torch.nn.Module, method: str, sample_input: torch.Tensor
    ):
        assert method in cls._shared_arg_indices, f"{method} is not supported"
        sample_output = module(sample_input)
        sample_input_size = sample_input.numel() / sample_input.shape[1]
        sample_output_size = sample_output.numel() / sample_output.shape[1]
        shared_arg_grans = [
            sample_input_size * ModuleDTypeSizeInByte[sample_input.dtype],
            sample_output_size * ModuleDTypeSizeInByte[sample_output.dtype],
        ]

        return cls._shared_arg_indices[method], shared_arg_grans

    @classmethod
    def extract_arg_infos(cls, module: torch.nn.Module, method: str):
        assert method in cls._shared_arg_indices, f"{method} is not supported"
        if not isinstance(module, torch.nn.Sequential):
            module = torch.nn.Sequential(module)
        module_name = cls.get_module_name(module)
        if module_name == "Conv2d" or module_name == "Conv2dReLU":
            input_arg_num = 2
        else:
            input_arg_num = 3
        total_arg_num = input_arg_num + 1

        return (
            input_arg_num,
            total_arg_num,
            cls._input_arg_indices[method],
            cls._output_arg_indices[method],
        )

    @classmethod
    def get_output_init_func(cls, module: torch.nn.Conv2d, method: str):
        conv_module = module[0] if isinstance(module, torch.nn.Sequential) else module
        assert isinstance(
            conv_module, torch.nn.Conv2d
        ), "An instance of Conv2d is expected"
        if method == "forward":

            def init_output(input):
                # TODO we can move shape calculation out of the function
                in_shape = input.shape
                if len(in_shape) == 3:
                    in_shape = (1,) + in_shape
                assert (
                    in_shape[1] == conv_module.in_channels
                ), "in_channels is not matched"
                out_n = in_shape[0]
                out_c = conv_module.out_channels
                out_h = (
                    in_shape[2]
                    + conv_module.padding[0] * 2
                    - conv_module.dilation[0] * (conv_module.kernel_size[0] - 1)
                    - 1
                ) // conv_module.stride[0] + 1
                out_w = (
                    in_shape[3]
                    + conv_module.padding[1] * 2
                    - conv_module.dilation[1] * (conv_module.kernel_size[1] - 1)
                    - 1
                ) // conv_module.stride[1] + 1
                out_shape = (
                    (out_n, out_c, out_h, out_w)
                    if len(in_shape) == 4
                    else (out_c, out_h, out_w)
                )
                return (torch.zeros(out_shape, dtype=input.dtype, device=input.device),)

        else:
            raise NotImplementedError(f"{method} is not supported")

        return init_output

    @classmethod
    def get_module_name(cls, module: torch.nn.Module) -> str:
        if not isinstance(module, torch.nn.Sequential):
            module = torch.nn.Sequential(module)
        for i, sub_module in enumerate(module):
            if i == 0:
                module_name = "Conv2d"
                if sub_module.bias is not None:
                    module_name += "Bias"
            elif isinstance(sub_module, cls._optional_succeed_module_cls):
                if isinstance(sub_module, torch.nn.BatchNorm2d):
                    module_name += "BatchNorm"
                    continue
                module_name += type(sub_module).__name__
        return module_name
