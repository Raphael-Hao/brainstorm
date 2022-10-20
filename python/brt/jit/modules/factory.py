# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch

from .base import ModuleInfo

class ModuleInfoFactory:
    @staticmethod
    def produce(module: torch.nn.Module) -> ModuleInfo:
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(module):
                return subclass(module)
        else:
            raise ValueError(f"Unknown module type: {module}")
        