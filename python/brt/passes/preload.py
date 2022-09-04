# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, Dict, List, Callable, Set

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from brt.passes.base import PassBase, register_pass
from brt.router import is_router
from brt.passes.utils import is_scatter, is_gather


@register_pass("preload")
class PreloadPass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        lazy_load_threshold=0.0,
        max_load_depth=0,
    ):
        super().__init__(m)
        self.lazy_load_threshold = lazy_load_threshold
        self.max_load_depth = max_load_depth

    def run_on_graph(self) -> None:
        sub_modules = dict(self.graph_mod.named_modules())
        memo = set()
        for node in self.graph_mod.graph.nodes:
            if node.op == "call_module":
                node_m = sub_modules[node.target]
                if is_scatter(node_m):
                    print(next(iter(node.users)).users)

    def travel_to_goal(
        self,
        start_nodes: List[Node],
        goal_classifiers: List[Callable],
        goal_nodes: Set[Node] = None,
        traveled_nodes: Set[Node] = None,
        current_depth=0,
        max_depth=0,
    ) -> List[Node]:
        # TODO add support to ignore call_function without computation: _operator.getitem
        assert (
            max_depth == 0
        ), f"Max_depth is set to {max_depth}, greater than 0 is not supported yet."
        if goal_nodes is None:
            goal_nodes = set()
        if traveled_nodes is None:
            traveled_nodes = set()
        nodes_to_travel: List[Node] = list()
        parameters_dict, buffers_dict = {}, {}
        for node in start_nodes:
            if any(goal_classifier(node) for goal_classifier in goal_classifiers):
                goal_nodes.add(node)
                continue
            if node not in traveled_nodes:
                if node.op in ["call_module", "call_function", "call_method"]:
                    traveled_nodes.add(node)
                if node.op == "call_module":
                    (
                        collected_params,
                        collected_buffers,
                    ) = self.get_module_params_or_buffers(node)
                    parameters_dict.update(collected_params)
                    buffers_dict.update(collected_buffers
                for user in node.users:
                    if user not in traveled_nodes:
                        nodes_to_travel.append(user)

    def get_module_params_or_buffers(self, node):
        sub_modules = dict(self.graph_mod.named_modules())
        node_m = sub_modules[node.target]
        parameter_dict = dict(node_m.named_parameters())
        buffer_dict = {}
        for bname, btensor in node_m.named_buffers():
            b_module_name, b_tensor_name = bname.rsplit(".", 1)
            b_owner_m = node_m.get_submodule(b_module_name)
            buffer_dict[bname] = (btensor, b_owner_m, b_tensor_name)
        return parameter_dict, buffer_dict

    def finalize(self) -> GraphModule:
        return super().finalize()
