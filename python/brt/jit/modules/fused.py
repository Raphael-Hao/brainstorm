from abc import abstractmethod
from typing import List, Tuple, Union, Literal, Callable

import torch
from torch import nn
from torch import autograd

from brt.jit.codegen.cuda import GlobalKernel
from brt.jit.modules.base import ModuleBase
from brt.jit.modules.atom import AtomModule
from brt.jit.modules import factory


class FusedModule(ModuleBase):
    def __init__(self, module: nn.ModuleList):
        super().__init__(module)
        assert isinstance(
            self.module, torch.nn.ModuleList
        ), "modules must be a ModuleList for fusion"
        self.num_submodule = len(self.module)
        self.jit_submodules: List[AtomModule] = [
            factory.JitModuleFactory.produce(submodule, opt_level=None)
            for submodule in self.module
        ]

    def _get_output_shape(
        self, method: str, sample_inputs: List[torch.Tensor]
    ) -> List[torch.Size]:
        return sum(
            [
                jsm._get_output_shape(method, sample_input)
                for jsm, sample_input in zip(self.jit_submodules, sample_inputs)
            ],
            start=[],
        )
        # return [
        #     jsm._get_output_shape(method, sample_input)
        #     for jsm, sample_input in zip(self.jit_submodules, sample_inputs)
        # ]
