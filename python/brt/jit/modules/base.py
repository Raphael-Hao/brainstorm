# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union

import torch

from brt.runtime import log
from brt.jit.codegen.module import ModuleKernel

logger = log.get_logger(__file__)

__all__ = ["ModuleInfo"]


class ModuleInfo:
    _input_arg_indices = None
    _output_arg_indices = None
    _shared_arg_indices = None

    def __init__(self, module: torch.nn.Module):
        assert type(self).ismodule(module)
        self.module = module
        self.module_name = self._get_module_name()

    @classmethod
    def ismodule(cls, module: torch.nn.Module) -> bool:
        raise NotImplementedError()

    def make_kernel(
        self,
        method: str,
        sample_input: torch.Tensor,
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> ModuleKernel:
        raise NotImplementedError()

    def extract_shared_arg_infos(
        self, method: str, sample_input: torch.Tensor,
    ) -> Tuple[List, List]:
        raise NotImplementedError()

    def extract_arg_infos(self, method: str,) -> Tuple[int, int, List, List]:
        raise NotImplementedError()

    def extract_parameters(self) -> List[torch.nn.Parameter]:
        raise NotImplementedError()

    def get_output_shape(self, method: str, sample_input: torch.Tensor) -> torch.Size:
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        return self.module.__getattr__(method)(sample_input).shape

    def _get_module_name(self) -> str:
        raise NotImplementedError()