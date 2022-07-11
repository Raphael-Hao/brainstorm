# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import brt
import torch
import torch.fx as fx
from brt.routers import is_router


class GraphTracer(fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if is_router(m):
            return True
        return super().is_leaf_module(m, module_qualified_name)
