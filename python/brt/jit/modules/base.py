# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Literal, Callable, Type, Dict

import torch
from torch import nn
from torch import autograd

from brt.runtime import log, BRT_KERNEL_TEMPLATE_PATH
from brt.jit.codegen.cuda import GlobalKernel
from brt.jit.compiler import CUDACompiler
from brt.trace.leaf_node import register_leaf_node

logger = log.get_logger(__file__)

AtomModuleInputType = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
FuseModuleInputType = List[AtomModuleInputType]
ModuleInputType = Union[AtomModuleInputType, FuseModuleInputType]


class ModuleBase(ABC):
    def __init__(self, module: nn.Module):
        self.module = module

    def make_kernel(
        self,
        sample_inputs: ModuleInputType,
        method: str = "forward",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> Callable[..., None]:
        global_kernel = self._make_global_kernel(
            sample_inputs=sample_inputs,
            method=method,
            objective_func=objective_func,
            rank=rank,
        )
        kernel_code, _, _, _ = global_kernel.get_code()
        processed_template_fname = str(
            BRT_KERNEL_TEMPLATE_PATH
            / ("processed_" + global_kernel.func_name[:10] + ".cu")
        )
        with open(processed_template_fname, "w") as f:
            f.write(kernel_code)
        jit_kernel = CUDACompiler.generate_kernel(None, kernel_code)
        return jit_kernel

    @abstractmethod
    def _make_global_kernel(
        self,
        sample_inputs: ModuleInputType,
        method: str = "forward",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> GlobalKernel:
        raise NotImplementedError()

    @abstractmethod
    def make_function(
        self,
        sample_inputs: ModuleInputType,
        mode: Literal["eval", "train"] = "eval",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> autograd.Function:
        raise NotImplementedError()

    # @abstractmethod
    def make_module(
        self,
        sample_inputs: ModuleInputType,
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def _extract_shared_arg_infos(
        self,
        method: str,
        sample_inputs: ModuleInputType,
    ) -> Tuple[List, List]:
        raise NotImplementedError()

    @abstractmethod
    def _extract_arg_infos(
        self,
        method: str,
    ) -> Tuple[int, int, List[int], List[int]]:
        raise NotImplementedError()

    @abstractmethod
    def _get_output_shape(
        self,
        method: str,
        sample_inputs: ModuleInputType,
    ) -> List[torch.Size]:
        raise NotImplementedError

    @property
    @abstractmethod
    def module_name(self) -> str:
        raise NotImplementedError()


@register_leaf_node
class JitModuleBase(nn.Module):
    def __init__(
        self,
        function: Type[autograd.Function],
        module_name: str = "BRT.Module",
        extra_repr: str = "",
        parameters: Dict[str, torch.Tensor] = {},
    ):
        super().__init__()
        self.function = function
        self._factory_cls = ModuleBase
        self._module_name = module_name
        self._extra_repr = extra_repr
        for param_name, param in parameters.items():
            self.register_parameter(param_name, param)

    @abstractmethod
    def forward(self):
        pass

    def _get_name(self):
        return self._module_name

    def extra_repr(self):
        return self._extra_repr
