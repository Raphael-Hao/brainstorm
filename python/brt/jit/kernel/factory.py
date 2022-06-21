# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Dict, List, Union

import torch
import torch.nn.functional as F

from .hetero_fused import HeteroFusedKernel
from .homo_fused import HomoFusedKernel
from .module import ModuleKernel
from .utils import to_list

__all__ = [
    "HeteroFusedKernelFactory",
    "HomoFusedKernelFactory",
    "ModuleKernelFactory",
]


class HeteroFusedKernelFactory:
    @staticmethod
    def make_kernel(modules: torch.nn.ModuleList, method, sample_inputs):
        candidates = []
        assert len(modules) == len(
            sample_inputs
        ), "modules and sample_inputs must have the same length"
        for m, sample_input in zip(modules, sample_inputs):
            module_kernel = ModuleKernelFactory.make_kernel(m, method, sample_input)
            candidates.append(module_kernel)
        fused_func = HeteroFusedKernel(candidates)
        return fused_func


class HomoFusedKernelFactory:
    @staticmethod
    def make_kernel(module: torch.nn.Module, method, sample_inputs):
        module_kernel = ModuleKernelFactory.make_kernel(module, method, sample_inputs)
        fused_func = HomoFusedKernel(module_kernel)
        return fused_func


class ModuleKernelFactory:
    @staticmethod
    def make_kernel(module: torch.nn.Module, method, sample_input) -> ModuleKernel:
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(module):
                return subclass.make_kernel(module, method, sample_input)
        raise ValueError(f"Unknown module type: {module}")

    @staticmethod
    def make_kernels(
        module: torch.nn.Module, method, sample_inputs
    ) -> List[ModuleKernel]:
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(module):
                ret_kernels = []
                for sample_input in sample_inputs:
                    ret_kernels.append(
                        subclass.make_kernel(module, method, sample_input)
                    )
                return ret_kernels
        raise ValueError(f"Unknown module type: {module}")


class ModuleInfo:
    __module_cls__ = None  # type: torch.nn.Module

    @classmethod
    def ismodule(cls, module: torch.nn.Module) -> bool:
        raise NotImplementedError()

    @classmethod
    def make_kernel(
        cls,
        module: torch.nn.Module,
        method: str,
        sample_input: Union[torch.Tensor, List[torch.Tensor]],
    ):
        raise NotImplementedError()

    @classmethod
    def get_argument_infos(cls, module: torch.nn.Module):
        raise NotImplementedError()


class LinearInfo(ModuleInfo):
    __module_cls__ = torch.nn.Linear
    __shared_arg__ = [0, 2]

    @classmethod
    def ismodule(cls, module: torch.nn.Module):
        return isinstance(module, cls.__module_cls__)

    @classmethod
    def make_kernel(
        cls,
        module: torch.nn.Linear,
        method: str,
        sample_input: torch.Tensor,
    ):
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
    def get_argument_infos(cls, module: torch.nn.Module):
        pass


class Conv2dBNActInfo(ModuleInfo):
    __module_cls__ = [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU]
    __shared_arg__ = [
        0,
    ]

    @classmethod
    def ismodule(cls, module: torch.nn.Module):
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
    def make_kernel(cls, module: torch.nn.Module, method, sample_input):
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

        sample_output = module(sample_input)
        input_infos = {"input_0": list(sample_input.shape)}
        output_infos = {"output_0": list(sample_output.shape)}

        return ModuleKernel(
            module_name=module_name,
            method=method,
            kernel_source=None,
            input_infos=input_infos,
            output_infos=output_infos,
            parameters=parameters,
        ).load_from_db()

    @classmethod
    def get_argument_infos(cls, module: torch.nn.Module):
        pass