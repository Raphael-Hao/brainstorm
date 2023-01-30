from abc import abstractmethod
from typing import List, Tuple, Union, Literal, Callable, Type, Dict

import torch
from torch import nn
from torch import autograd

from brt.jit.codegen.cuda import GlobalKernel
from brt.jit.modules.base import ModuleBase, JitModuleBase
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


class JitFusedModule(JitModuleBase):
    def __init__(
        self,
        function: Type[autograd.Function],
        module_name: str = "BRT.FusedModule",
        extra_repr: str = "",
        parameters: Dict[str, torch.Tensor] = ...,
    ):
        super().__init__(function, module_name, extra_repr, parameters)
        self._factory_cls = FusedModule
