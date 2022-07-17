# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.fx as fx
from brt.router import is_router
from brt.trace.leaf_node import is_leaf_node


class GraphTracer(fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if is_router(m) or is_leaf_node(m):
            return True
        return super().is_leaf_module(m, module_qualified_name)
