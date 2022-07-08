# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import brt
import torch
import torch.fx as fx


class BRTTRacer(fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, brt.routers.RouterBase):
            return True
        return super().is_leaf_module(m, module_qualified_name)
