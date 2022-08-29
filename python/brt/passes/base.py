# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.fx.graph_module import GraphModule
from brt.runtime import Registry
from brt.trace import symbolic_trace


class PassBase:
    def __init__(self, m: torch.nn.Module):
        self.graph_mod = symbolic_trace(m)

    def analyze(flow: torch.Tensor) -> None:
        raise NotImplementedError

    def ru_on_graph(self, graph) -> None:
        raise NotImplementedError

    def finalize(self) -> GraphModule:
        self.graph_mod.graph.eliminate_dead_code()
        self.graph_mod.recompile()
        return self.graph_mod


def register_pass(pass_class: type) -> None:
    return Registry.register_sub_cls(pass_class, PassBase)
