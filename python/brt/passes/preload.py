# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, Dict

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from brt.passes.base import PassBase, register_pass
from brt.router import is_router
from brt.passes.utils import is_scatter, is_gather


@register_pass("preload")
class PreloadPass(PassBase):
    def __init__(self, m: Union[torch.nn.Module, GraphModule], lazy_load_threshold=0.0):
        super().__init__(m)

    def run_on_graph(self) -> None:
        sub_modules = dict(self.graph_mod.named_modules())
        memo = set()
        for node in self.graph_mod.graph.nodes:
            if node.op == "call_module":
                node_m = sub_modules[node.target]
                if is_scatter(node_m):
                    # self.collect_parameter_and_buffers(node, sub_modules)
                    print(next(iter(node.users)).users)

    def collect_parameter_and_buffers(
        self, r_node: Node, sub_modules: Dict[str, nn.Module], path_num=0
    ):
        parameter_dict = {} if path_num == 0 else [{} for _ in range(path_num)]
        buffer_dict = {} if path_num == 0 else [{} for _ in range(path_num)]
        node_user_num = len(r_node.users)
        assert path_num == 0 or path_num == node_user_num, "path_num should be 0 or equal to node_user_num"

        def travel_to_routers(
            node: Node, sub_modules: Dict[str, nn.Module], router_nodes=None
        ):
            pass

        for user in r_node.users:
            print(user.target)

    def finalize(self) -> GraphModule:
        return super().finalize()
