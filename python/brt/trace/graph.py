# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.fx as fx
from brt.router import is_router
from brt.trace.leaf_node import is_leaf_node

__all__ = ["symbolic_trace"]


class GraphTracer(fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        
        ##FIXME this is error when we use deepcopy is_router always returns false
        if is_router(m) or is_leaf_node(m):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def symbolic_trace(m: nn.Module, name=None) -> fx.GraphModule:
    assert isinstance(
        m, nn.Module
    ), "brt provided symbolic_trace only works on nn.Modules"
    tracer = GraphTracer()
    graph = tracer.trace(m)
    name = m.__class__.__name__ if name is None else name
    return fx.GraphModule(tracer.root, graph, name)
