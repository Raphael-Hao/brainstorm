# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple, Union

import torch

from ..kernel.module import ModuleDTypeSizeInByte, ModuleKernel


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
    ) -> ModuleKernel:
        raise NotImplementedError()

    @classmethod
    def extract_shared_arg_infos(
        cls,
        module: torch.nn.Module,
        method: str,
        sample_input: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[List, List]:
        raise NotImplementedError()

    @classmethod
    def extract_arg_infos(
        cls,
        module: torch.nn.Module,
        method: str,
    ) -> Tuple[int, int, List, List]:
        raise NotImplementedError()

    @classmethod
    def get_output_init_func(cls, module: torch.nn.Module, method: str):
        raise NotImplementedError()


class LinearInfo(ModuleInfo):
    __module_cls__ = torch.nn.Linear
    __input_arg_indices__ = {"forward": [0]}
    __parameter_arg_indices__ = {"forward": [1]}
    __output_arg_indices__ = {"forward": [2]}
    __shared_arg_indices__ = {"forward": [0, 2]}

    @classmethod
    def ismodule(cls, module: torch.nn.Module):
        return isinstance(module, cls.__module_cls__)

    @classmethod
    def make_kernel(
        cls, module: torch.nn.Linear, method: str, sample_input: torch.Tensor
    ) -> ModuleKernel:
        assert method in cls.__shared_arg_indices__, f"{method} is not supported"
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
        assert method in cls.__shared_arg_indices__, f"{method} is not supported"
        shared_arg_grans = [
            module.in_features * ModuleDTypeSizeInByte[sample_input.dtype],
            module.out_features * ModuleDTypeSizeInByte[sample_input.dtype],
        ]
        return cls.__shared_arg_indices__[method], shared_arg_grans

    @classmethod
    def extract_arg_infos(cls, module: torch.nn.Linear, method: str):
        assert method in cls.__shared_arg_indices__, f"{method} is not supported"
        if method.bias is None:
            input_arg_num = 2
        else:
            input_arg_num = 3
        total_arg_num = input_arg_num + 1
        return (
            input_arg_num,
            total_arg_num,
            cls.__input_arg_indices__[method],
            cls.__output_arg_indices__[method],
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


class Conv2dBNActInfo(ModuleInfo):
    __module_cls__ = [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU]
    __input_arg_indices__ = {"forward": [0]}
    __parameter_arg_indices__ = {"forward": [1]}
    __output_arg_indices__ = {"forward": [2]}
    __shared_arg_indices__ = {"forward": [0, 2]}

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
    def make_kernel(
        cls, module: torch.nn.Module, method: str, sample_input: torch.Tensor
    ) -> ModuleKernel:
        assert method in cls.__shared_arg_indices__, f"{method} is not supported"
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
        cls, module: torch.nn.Module, method: str, sample_input: torch.Tensor
    ):
        assert method in cls.__shared_arg_indices__, f"{method} is not supported"
        sample_output = module(sample_input)
        sample_input_size = sample_input.numel() / sample_input.shape[1]
        sample_output_size = sample_output.numel() / sample_output.shape[1]
        shared_arg_grans = [
            sample_input_size * ModuleDTypeSizeInByte[sample_input.dtype],
            sample_output_size * ModuleDTypeSizeInByte[sample_output.dtype],
        ]

        return cls.__shared_arg_indices__[method], shared_arg_grans

    @classmethod
    def extract_arg_infos(cls, module: torch.nn.Module, method: str):
        assert method in cls.__shared_arg_indices__, f"{method} is not supported"
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
            cls.__input_arg_indices__[method],
            cls.__output_arg_indices__[method],
        )

    @classmethod
    def get_output_init_func(cls, module: torch.nn.Conv2d, method: str):
        conv_module = module[0] if isinstance(module, torch.nn.Sequential) else module
        assert isinstance(
            conv_module, torch.nn.Conv2d
        ), f"An instance of Conv2d is expected"
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
    def get_module_name(cls, modules) -> str:
        module_name = "Conv2d"
        module_name += "Bias" if modules[0].bias is not None else ""
        if len(modules) == 2:
            module_name += (
                "BatchNorm" if isinstance(modules[1], torch.nn.BatchNorm2d) else "ReLU"
            )
        elif len(modules) == 3:
            module_name += "BatchNormReLU"
        return module_name
