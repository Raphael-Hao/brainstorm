# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.jit.modules.base import ModuleInfo
from brt.jit.codegen.module import ModuleKernel, ModuleDTypeSizeInByte

logger = log.get_logger(__file__)


class ConvTranspose2dInfo(ModuleInfo):
    """Info for torch.nn.ConvTranspose2d kernel

    Method Args:
        forward:
            [0]: float* __restrict__ placeholder,  Input Tensor
            [1]: float* __restrict__ placeholder1, ConvTranspose2d.weight
            [2]: float* __restrict__ T_add,        Output Tensor
            [3]: float* __restrict__ placeholder2, ConvTranspose2d.bias
    """

    _involved_module_cls = torch.nn.ConvTranspose2d
    _input_arg_indices = {"forward": [0]}
    _output_arg_indices = {"forward": [2]}
    _shared_arg_indices = {"forward": [0, 2]}

    @classmethod
    def ismodule(cls, module: torch.nn.ConvTranspose2d):
        return isinstance(module, cls._involved_module_cls)

    @classmethod
    def make_kernel(
        cls, module: torch.nn.ConvTranspose2d, method: str, sample_input: torch.Tensor
    ) -> ModuleKernel:
        assert method in cls._shared_arg_indices, f"{method} is not supported"
        module_name = cls.get_module_name(module)
        parameters = {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
            "output_padding": module.output_padding[0]
            if module.output_padding[0] == module.output_padding[1]
            else module.output_padding[0],
        }
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
        ).load_from_db()

    @classmethod
    def extract_shared_arg_infos(
        cls, module: torch.nn.ConvTranspose2d, method: str, sample_input: torch.Tensor
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
    def extract_arg_infos(cls, module: torch.nn.ConvTranspose2d, method: str):
        assert method in cls._shared_arg_indices, f"{method} is not supported"
        if module.bias is None:
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
    def get_output_init_func(cls, module: torch.nn.ConvTranspose2d, method: str):
        raise NotImplementedError("TODO")

    @classmethod
    def get_module_name(cls, modules: torch.nn.ConvTranspose2d) -> str:
        if modules.bias is None:
            module_name = "ConvTranspose2d"
        else:
            module_name = "ConvTranspose2dBias"
        return module_name
