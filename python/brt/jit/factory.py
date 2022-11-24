from typing import List, Tuple, Union, Literal, Callable

import torch
from torch import autograd
from torch import nn

from brt.runtime import BRT_KERNEL_TEMPLATE_PATH
from brt.jit.modules import JitModuleFactory
from brt.jit.compiler import CUDACompiler
from brt.jit.modules import (
    JitModuleFactory,
    ModuleBase,
    HorizFusedModule,
    HeteroFusedModule,
    HomoFusedModule,
)

# from ._kernel import make_jit_kernel as _make_jit_kernel

__all__ = [
    "make_jit_kernel",
    "make_jit_function",
    "make_jit_module",
    "ModuleFactory",
]


def make_jit_kernel(
    modules: Union[nn.Module, nn.ModuleList],
    sample_inputs,
    method: str = "forward",
    opt_level: Union[str, None] = None,
    objective_func: str = "fastest",
    rank: Union[int, List[int]] = 1,
) -> Callable[..., None]:
    return ModuleFactory.make_kernel(
        modules,
        sample_inputs=sample_inputs,
        method=method,
        opt_level=opt_level,
        objective_func=objective_func,
        rank=rank,
    )


def make_jit_function(
    modules: Union[nn.Module, nn.ModuleList],
    sample_inputs,
    mode: Literal["eval", "train"] = "eval",
    opt_level: Union[str, None] = None,
    objective_func: str = "fastest",
    rank: Union[int, List[int]] = 1,
) -> autograd.Function:
    return ModuleFactory.make_function(
        modules,
        sample_inputs=sample_inputs,
        mode=mode,
        opt_level=opt_level,
        objective_func=objective_func,
        rank=rank,
    )


def make_jit_module(
    modules: Union[nn.Module, nn.ModuleList],
    sample_inputs,
    opt_level: Union[str, None] = None,
    objective_func: str = "fastest",
    rank: Union[int, List[int]] = 1,
) -> nn.Module:
    return ModuleFactory.make_module(
        modules,
        sample_inputs=sample_inputs,
        opt_level=opt_level,
        objective_func=objective_func,
        rank=rank,
    )


class ModuleFactory:
    @staticmethod
    def make_kernel(
        modules,
        sample_inputs,
        method="forward",
        opt_level=None,
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> Callable[..., None]:
        jit_module = JitModuleFactory.produce(modules, opt_level)
        return jit_module.make_kernel(
            sample_inputs=sample_inputs,
            method=method,
            objective_func=objective_func,
            rank=rank,
        )

    @staticmethod
    def make_function(
        modules,
        sample_inputs,
        mode: Literal["eval", "train"] = "eval",
        opt_level=None,
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> autograd.Function:
        jit_module = JitModuleFactory.produce(modules, opt_level)
        return jit_module.make_function(
            sample_inputs=sample_inputs,
            mode=mode,
            objective_func=objective_func,
            rank=rank,
        )

    @staticmethod
    def make_module(
        modules,
        sample_inputs,
        opt_level=None,
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> nn.Module:
        jit_module = JitModuleFactory.produce(modules, opt_level)
        return jit_module.make_module(
            sample_inputs=sample_inputs,
            objective_func=objective_func,
            rank=rank,
        )
