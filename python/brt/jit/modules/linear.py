# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.jit.modules.base import ModuleInfo
from brt.jit.codegen.module import ModuleKernel, ModuleDTypeSizeInByte

logger = log.get_logger(__file__)


class LinearInfo(ModuleInfo):
    _involved_module_cls = torch.nn.Linear
    _input_arg_indices = {"forward": [0]}
    _parameter_arg_indices = {"forward": [1]}
    _output_arg_indices = {"forward": [2]}
    _shared_arg_indices = {"forward": [0, 2]}

    @classmethod
    def ismodule(cls, module: torch.nn.Module):
        return isinstance(module, cls.__module_cls__)

    @classmethod
    def make_kernel(
        cls, module: torch.nn.Linear, method: str, sample_input: torch.Tensor
    ) -> ModuleKernel:
        assert method in cls._shared_arg_indices, f"{method} is not supported"
        module_name = "Linear" if module.bias is None else "LinearBias"
        sample_output = module(sample_input)
        input_infos = {"input_0": list(sample_input.shape)}
        output_infos = {"output_0": list(sample_output.shape)}
        parameters = {
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
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
        cls, module: torch.nn.Linear, method: str, sample_input: torch.Tensor
    ):
        assert method in cls._shared_arg_indices, f"{method} is not supported"
        shared_arg_grans = [
            module.in_features * ModuleDTypeSizeInByte[sample_input.dtype],
            module.out_features * ModuleDTypeSizeInByte[sample_input.dtype],
        ]
        return cls._shared_arg_indices[method], shared_arg_grans

    @classmethod
    def extract_arg_infos(cls, module: torch.nn.Linear, method: str):
        assert method in cls._shared_arg_indices, f"{method} is not supported"
        if method.bias is None:
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
    def get_output_init_func(cls, module: torch.nn.Linear, method: str):
        if method == "forward":

            def init_output(input):
                # TODO we can move shape calculation out of the function
                in_shape = input.shape
                assert in_shape[-1] == module.in_features, "in_features is not matched"
                out_shape = in_shape[:-1] + (module.out_features,)
                return (torch.zeros(out_shape, dtype=input.dtype, device=input.device),)

        else:
            raise NotImplementedError(f"{method} is not supported")
        return init_output
