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
        self.goal_classifiers: List[Callable] = []
        self.all_parameters_memo: Set[torch.nn.Parameter] = set()
        self.all_buffers_memo: Set[torch.Tensor] = set()

        assert (
            self.max_load_depth == 0
        ), f"Max_depth is set to {self.max_load_depth}, greater than 0 is not supported yet."

    def _build_goal_classifiers(self) -> None:
        self.goal_classifiers = [
            is_router,
            is_scatter,
            is_gather,
        ]

    def run_on_graph(self) -> None:
        """TODO
        1. find the first part of parameters and buffers to be preloaded.
        2. insert preloading, guarding, unloading hook to enable the pipeline.
        """
        sub_modules = dict(self.graph_mod.named_modules())
        memo = set()
        for node in self.graph_mod.graph.nodes:
            if node.op == "call_module":
                node_m = sub_modules[node.target]
                if is_scatter(node_m):
                    print(next(iter(node.users)).users)

    def travel_to_goal(self, start_nodes: List[Node]) -> List[Node]:
        """TODO
        1. get the first module to insert preloading hook and weight ready guard.
        2. return the buffers and parameters to be preloaded.
        """
        goal_nodes = set()
        traveled_nodes = set()
        parameters_dict, buffers_dict = self._travel_to_goal(
            start_nodes=start_nodes,
            goal_nodes=goal_nodes,
            traveled_nodes=traveled_nodes,
            current_depth=0,
        )
        return goal_nodes, traveled_nodes, parameters_dict, buffers_dict

    def _travel_to_goal(
        self,
        start_nodes: List[Node],
        goal_nodes: Set[Node] = None,
        traveled_nodes: Set[Node] = None,
        current_depth=0,
    ) -> List[Node]:
        """TODO
        1. check whether the parameters and buffers are already in the memo.
        2. add support to ignore call_function without computation: _operator.getitem
        """
        if goal_nodes is None:
            goal_nodes = set()
        if traveled_nodes is None:
            traveled_nodes = set()
        nodes_to_travel: List[Node] = list()
        parameters_dict, buffers_dict = {}, {}

        for node in start_nodes:
            if self.max_load_depth > 0 and current_depth >= self.max_load_depth:
                goal_nodes.add(node)
                continue
            if any(goal_classifier(node) for goal_classifier in self.goal_classifiers):
                goal_nodes.add(node)
                continue
            if node not in traveled_nodes:
                if node.op in ["call_module", "call_function", "call_method"]:
                    traveled_nodes.add(node)
                    if node.op == "call_module":
                        (
                            collected_params,
                            collected_buffers,
                        ) = self._get_module_params_or_buffers(node)
                        parameters_dict.update(collected_params)
                        buffers_dict.update(collected_buffers)
                    else:
                        (
                            collected_params,
                            collected_buffers,
                        ) = self._get_function_method_params_or_buffers(node)
                        parameters_dict.update(collected_params)
                        buffers_dict.update(collected_buffers)
                for user in node.users:
                    if user not in traveled_nodes:
                        nodes_to_travel.append(user)

        if len(nodes_to_travel) > 0:
            collected_params, collected_buffers = self._travel_to_goal(
                start_nodes=nodes_to_travel,
                goal_nodes=goal_nodes,
                traveled_nodes=traveled_nodes,
                current_depth=current_depth + 1,
            )
            parameters_dict.update(collected_params)
            buffers_dict.update(collected_buffers)

        return parameters_dict, buffers_dict

    def _get_module_params_or_buffers(self, node: Node):
        sub_modules = dict(self.graph_mod.named_modules())
        node_m = sub_modules[node.target]
        parameter_dict = dict(node_m.named_parameters(node.target))
        buffer_dict = {}
        for bname, btensor in node_m.named_buffers(node.target):
            b_module_name, _, b_tensor_name = bname.rpartition(".")
            b_owner_m = self.graph_mod.get_submodule(b_module_name)
            buffer_dict[bname] = (btensor, b_owner_m, b_tensor_name)
        return parameter_dict, buffer_dict

    def _get_function_method_params_or_buffers(self, node: Node):
        parameter_dict = {}
        buffer_dict = {}
        for in_node in node.all_input_nodes:
            if in_node.op is "get_attr":
                attr_module_name, _, attr_name = in_node.target.rpartition(".")
                attr_owner_module = self.graph_mod.get_submodule(attr_module_name)

                assert hasattr(attr_owner_module, attr_name), "Attribute not found."

                attr_v = getattr(attr_owner_module, attr_name)
                if isinstance(attr_v, torch.nn.Parameter):
                    parameter_dict[in_node.target] = attr_v
                elif attr_name in attr_owner_module._buffers:
                    buffer_dict[in_node.target] = (attr_v, attr_owner_module, attr_name)
        return parameter_dict, buffer_dict

    def finalize(self) -> GraphModule:
        return super().finalize()
