# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union

import torch
import torch.nn.functional as F

from .module import ModuleFunction
from .utils import to_list


class HeteroFusedFunctionFactory:
    @staticmethod
    def make_module_function(module: torch.nn.Module, method, sample_inputs):
        for subclass in ModuleInfo.__subclasses__():
            if subclass._ismodule(module):
                return subclass._make_module_functions(module, method, sample_inputs)
        raise ValueError(f"Unknown module type: {module}")

class HomoFusedFunctionFactory:
    @staticmethod
    def make_module_function(module: torch.nn.Module, method, sample_inputs):
        for subclass in ModuleInfo.__subclasses__():
            if subclass._ismodule(module):
                return subclass._make_module_functions(module, method, sample_inputs)
        raise ValueError(f"Unknown module type: {module}")


class ModuleFunctionFactory:
    @staticmethod
    def make_module_functions(module: torch.nn.Module, method, sample_inputs):
        for subclass in ModuleInfo.__subclasses__():
            if subclass._ismodule(module):
                return subclass._make_module_functions(module, method, sample_inputs)
        raise ValueError(f"Unknown module type: {module}")


class ModuleInfo:
    @classmethod
    def _ismodule(cls, module: torch.nn.Module) -> bool:
        raise NotImplementedError()

    @classmethod
    def _make_module_functions(
        cls,
        module: torch.nn.Module,
        method: str,
        sample_inputs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[ModuleFunction]:
        raise NotImplementedError()


class LinearInfo(ModuleInfo):
    __module_cls__ = torch.nn.Linear
    __shared_arg__ = [0, 2]

    @classmethod
    def _ismodule(cls, module: torch.nn.Module):
        return isinstance(module, cls.__module_cls__)

    @classmethod
    def _make_module_functions(
        cls,
        module: torch.nn.Linear,
        method: str,
        sample_inputs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[ModuleFunction]:
        module_name = "Linear" if module.bias is None else "LinearBias"
        sample_inputs = to_list(sample_inputs)
        ret_functions = []
        for sample_input in sample_inputs:
            sample_output = module(sample_input)
            input_infos = {"input_0": list(sample_input.shape)}
            output_infos = {"output_0": list(sample_output.shape)}
            parameters = {
                "in_features": module.in_features,
                "out_features": module.out_features,
            }
            ret_functions.append(
                ModuleFunction(
                    module_name=module_name,
                    method=method,
                    kernel_source=None,
                    input_infos=input_infos,
                    output_infos=output_infos,
                    parameters=parameters,
                )
            )
        return ret_functions


class Conv2dBNActInfo(ModuleInfo):
    __module_cls__ = [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU]
    __shared_arg__ = [
        0,
    ]

    @classmethod
    def _ismodule(cls, module: torch.nn.Module):
        if not isinstance(module, torch.nn.Sequential):
            module = torch.nn.Sequential(module)
        if len(module) == 1 and isinstance(module[0], cls.__module_cls__[0]):
            return True
        if len(module) == 2 and isinstance(module[0], cls.__module_cls__[0]):
            if isinstance(module[1], cls.__module_cls__[1]) or isinstance(
                module[1], cls.__module_cls__[2]
            ):
                return True
        if (
            len(module) == 3
            and isinstance(module[0], cls.__module_cls__[0])
            and isinstance(module[1], cls.__module_cls__[1])
            and isinstance(module[2], cls.__module_cls__[2])
        ):
            return True
        return False

    @classmethod
    def _make_module_function(cls, module: torch.nn.Module, method, sample_inputs):
        if not isinstance(module, torch.nn.Sequential):
            module = torch.nn.Sequential(module)

        module_name = "Conv2d"
        module_name += "Bias" if module[0].bias is not None else ""
        parameters = {}
        parameters["in_channels"] = module[0].in_channels
        parameters["out_channels"] = module[0].out_channels
        parameters["kernel_size"] = module[0].kernel_size
        parameters["stride"] = module[0].stride
        parameters["padding"] = module[0].padding
        parameters["dilation"] = module[0].dilation
        parameters["groups"] = module[0].groups
        if len(module) == 2:
            module_name += (
                "BatchNorm" if isinstance(module[1], torch.nn.BatchNorm2d) else "ReLU"
            )
        elif len(module) == 3:
            module_name += "BatchNormReLU"
        sample_inputs = to_list(sample_inputs)
        ret_functions = []

        for sample_input in sample_inputs:
            sample_output = module(sample_input)
            input_infos = {"input_0": list(sample_input.shape)}
            output_infos = {"output_0": list(sample_output.shape)}
            ret_functions.append(
                ModuleFunction(
                    module_name=module_name,
                    method=method,
                    kernel_source=None,
                    input_infos=input_infos,
                    output_infos=output_infos,
                    parameters=parameters,
                )
            )
        return ret_functions
