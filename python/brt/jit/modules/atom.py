# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List, Tuple, Union, Literal

import torch
from torch import nn

from brt.runtime import log
from brt.jit.modules.base import ModuleBase
from brt.jit.codegen.module import ModuleKernel

logger = log.get_logger(__file__)

__all__ = ["AtomModule"]


class AtomModule(ModuleBase):
    _input_arg_indices = None
    _output_arg_indices = None
    _shared_arg_indices = None

    def __init__(self, module: nn.Module):
        assert type(self).ismodule(module)
        super().__init__(module)

    @abstractmethod
    def _make_global_kernel(
        self,
        sample_inputs: Union[torch.Tensor, List[torch.Tensor]],
        method: str = "forward",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> ModuleKernel:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def ismodule(cls, module: nn.Module) -> bool:
        raise NotImplementedError()
