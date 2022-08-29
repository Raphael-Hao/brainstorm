# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import torch.nn as nn
from torch.fx.graph_module import GraphModule
from brt.trace.graph import GraphTracer

def symbolic_trace(m: nn.Module, name = None) -> GraphModule:
    assert isinstance(m, nn.Module), "brt provided symbolic_trace only works on nn.Modules"
    tracer = GraphTracer()
    graph = tracer.trace(m)
    name =m.__class__.__name__ if name is None else name
    return GraphModule(tracer.root, graph, name)
