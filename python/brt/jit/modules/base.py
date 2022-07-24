# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union

import torch
from brt.runtime import log
from brt.jit.codegen.module import ModuleKernel

logger = log.get_logger(__file__)

__all__ = ["ModuleInfo"]


class ModuleInfo:
    _involved_module_cls = None
    _input_arg_indices = None
    _parameter_arg_indices = None
    _output_arg_indices = None
    _shared_arg_indices = None

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
    def extract_parameters(cls, module) -> List[torch.nn.Parameter]:
        raise NotImplementedError()

    @classmethod
    def get_output_init_func(cls, module: torch.nn.Module, method: str):
        raise NotImplementedError()
