from abc import abstractmethod
from typing import List, Tuple, Union, Literal, Callable

import torch
from torch import nn
from torch import autograd

from brt.jit.codegen.cuda import GlobalKernel
from brt.jit.modules.base import ModuleBase
from brt.jit.modules import factory


class FusedModule(ModuleBase):
    def __init__(self, module: nn.ModuleList):
        super().__init__(module)
        self.jit_submodules: List[ModuleBase] = []
        for submodule in self.module:
            self.jit_submodules.append(
                factory.JitModuleFactory.produce(submodule, opt_level=None)
            )
