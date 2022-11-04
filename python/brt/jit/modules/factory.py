# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, List

import torch
from torch import nn

from brt.jit.modules.base import ModuleBase
from brt.jit.modules.atom import AtomModule
from brt.jit.modules.horiz import HorizFusedModule
from brt.jit.modules.hetero import HeteroFusedModule
from brt.jit.modules.homo import HomoFusedModule


class JitModuleFactory:
    @staticmethod
    def produce(
        modules: Union[nn.Module, nn.ModuleList], opt_level: Union[str, None] = None
    ) -> ModuleBase:
        if opt_level is None:
            for subclass in AtomModule.__subclasses__():
                if subclass.ismodule(modules):
                    return subclass(modules)
            else:
                raise ValueError(f"Unknown module type: {modules}")
        elif opt_level == "horiz_fuse":
            return HorizFusedModule(modules)
        elif opt_level == "hetero_fuse":
            return HeteroFusedModule(modules)
        elif opt_level == "homo_fuse":
            return HomoFusedModule(modules)
        else:
            raise ValueError(f"Not supported optimize level: {opt_level}")
