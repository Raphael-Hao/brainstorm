# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, Dict

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from brt.passes.base import PassBase, register_pass
from brt.router import is_router
from brt.passes.utils import is_scatter, is_gather


@register_pass("weight_preload")
class WeightPreloadPass(PassBase):
    def __init__(self, m: Union[torch.nn.Module, GraphModule]):
        super().__init__(m)

    def run_on_graph(self) -> None:
        sub_modules = dict(self.graph_mod.named_modules())
        for node in self.graph_mod.graph.nodes:
            if node.op == "call_module":
                node_m = sub_modules[node.target]
                if is_scatter(node_m):
                    self.collect_parameter_and_buffers(node, sub_modules)

    def collect_parameter_and_buffers(
        self, r_node: Node, sub_modules: Dict[str, nn.Module], depth: int = -1
    ):
        weight_dict = {}
        for user in r_node.users:
            print(user.target)
    def finalize(self) -> GraphModule:
        return super().finalize()
