# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import typing
from typing import Union, Dict, List, Callable, Set, Tuple, Any

from collections import OrderedDict
import re

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from brt.passes.base import PassBase, register_pass
from brt.passes.utils import debug_node
from brt.runtime.memory_planner import (
    MemoryPlanContext,
    OnDemandLoader,
    OnDemandUnloader,
    OnDemandGuarder,
    PredictLoader,
    PredictGuarder,
    PredictUnloader,
)

PLGT = List[
    Tuple[
        Node,
        Dict[Node, None],
        Dict[str, nn.Parameter],
        Dict[str, Tuple[nn.Module, str]],
    ]
]


@register_pass("MemoryPlan")
class MemoryPlanPass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        max_depth=0,
    ):
        super().__init__(m)
        self.max_depth = max_depth
        self.goal_classifiers: List[Callable] = []
        assert (
            self.max_depth == 0
        ), f"Max_depth is set to {self.max_depth}, greater than 0 is not supported yet."
        self.plan_groups: Dict[Node, PLGT] = dict()
        self.all_collected_params: Dict[str, torch.Tensor] = dict()
        self.all_collected_buffers: Dict[
            str, Tuple[torch.Tensor, nn.Module, str]
        ] = dict()

    def build_goal_classifiers(
        self, node_types: List[str] = None, custom_check_fns: List[Callable] = None
    ) -> None:
        self.goal_classifiers = []
        if node_types is None and custom_check_fns is None:
            return
        if node_types is not None:
            for node_type in node_types:
                node_type_check = "is_" + node_type + "_node"
                assert hasattr(
                    self, node_type_check
                ), f"Node type check for {node_type} is not supported internally."
                self.goal_classifiers.append(getattr(self, node_type_check))
        if custom_check_fns is not None:
            for custom_check_fn in custom_check_fns:
                self.goal_classifiers.append(custom_check_fn)

    def run_on_graph(self) -> None:
        self.collect_params_and_buffers()
        self.process_plan()

    def collect_params_and_buffers(self):
        self.build_goal_classifiers(node_types=["output", "scatter"])
        self.set_ignore_buffers([r"\.load\_history", r"\.capacity\_history"])

        memo = set()
        # first gropu will include the the placeholder and the first group of goal nodes (routers)

        start_nodes = self.find_all_placeholders()
        while start_nodes:
            (
                goal_nodes,
                traveled_nodes,
                collected_params,
                collected_buffers,
            ) = self.travel_to_goal(start_nodes=start_nodes, memo=memo)
            traveled_nodes = self.remove_placeholder(traveled_nodes)
            self.add_new_plan_group(traveled_nodes, collected_params, collected_buffers)
            start_nodes = self.remove_output(goal_nodes)
        assert self.is_params_and_buffer_all_collected()

    def process_plan(self):
        raise NotImplementedError

    def add_new_plan_group(
        self,
        traveled_nodes: Dict[Node, None],
        collected_params: Dict[str, nn.Parameter],
        collected_buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]],
        head_node: Node = None,
    ):
        first_node, last_node = self.find_first_and_last_node(traveled_nodes)
        head_node = first_node if head_node is None else head_node
        self.plan_groups.setdefault(head_node, []).append(
            (last_node, traveled_nodes, collected_params, collected_buffers)
        )
        self.all_collected_params.update(collected_params)
        self.all_collected_buffers.update(collected_buffers)

    def sort_plan_groups(self):
        orderred_plan_groups: typing.OrderedDict[Node, PLGT] = OrderedDict()
        for node in self.graph_mod.graph.nodes:
            if node in self.plan_groups:
                orderred_plan_groups[node] = self.plan_groups[node]
        return orderred_plan_groups

    def travel_to_goal(self, start_nodes: Dict[Node, None], memo: Set[Node] = None):
        """TODO
        1. get the first module to insert preloading hook and weight ready guard.
        2. return the buffers and parameters to be preloaded.
        """
        memo = set() if memo is None else memo
        (
            goal_nodes,
            traveled_nodes,
            collected_params,
            collected_buffers,
        ) = self._travel_to_goal(
            start_nodes=start_nodes,
            memo=memo,
            current_depth=0,
        )
        return goal_nodes, traveled_nodes, collected_params, collected_buffers

    def _travel_to_goal(
        self, start_nodes: Dict[Node, None], memo: Set[Node], current_depth=0
    ):
        """TODO
        1. check whether the parameters and buffers are already in the memo.
        2. add support to ignore call_function without computation: _operator.getitem
        """
        nodes_to_travel: Dict[Node, None] = {}
        goal_nodes: Dict[Node, None] = {}
        traveled_nodes: Dict[Node, None] = {}
        collected_params: Dict[str, nn.Parameter] = {}
        collected_buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]] = {}

        for node in start_nodes:
            if node not in memo:
                memo.add(node)
                traveled_nodes.setdefault(node)
                if self.is_module_node(node):
                    if not self.is_router_node(node):
                        (
                            params,
                            buffers,
                        ) = self._get_module_params_or_buffers(node)
                        collected_params.update(params)
                        collected_buffers.update(buffers)
                else:
                    (params, buffers,) = self._get_function_method_params_or_buffers(
                        node, traveled_nodes, memo
                    )
                    collected_params.update(params)
                    collected_buffers.update(buffers)

                for user in node.users:
                    if user not in memo:
                        if self.max_depth > 0 and current_depth >= self.max_depth:
                            goal_nodes.setdefault(user)
                            continue
                        if any(
                            goal_classifier(user)
                            for goal_classifier in self.goal_classifiers
                        ):
                            goal_nodes.setdefault(user)
                            continue
                        if user not in traveled_nodes:
                            nodes_to_travel.setdefault(user)

        if len(nodes_to_travel) > 0:
            (
                _goal_nodes,
                _traveled_nodes,
                _collected_params,
                _collected_buffers,
            ) = self._travel_to_goal(
                start_nodes=nodes_to_travel,
                memo=memo,
                current_depth=current_depth + 1,
            )
            goal_nodes.update(_goal_nodes)
            traveled_nodes.update(_traveled_nodes)
            collected_params.update(_collected_params)
            collected_buffers.update(_collected_buffers)

        return goal_nodes, traveled_nodes, collected_params, collected_buffers

    def _get_module_params_or_buffers(self, node: Node):
        node_m = self.sub_modules[node.target]
        parameter_dict = dict(node_m.named_parameters(node.target))
        buffer_dict = {}
        for bname, btensor in node_m.named_buffers(node.target):
            b_module_name, _, b_tensor_name = bname.rpartition(".")
            b_owner_m = self.graph_mod.get_submodule(b_module_name)
            buffer_dict[bname] = (btensor, b_owner_m, b_tensor_name)
        return parameter_dict, buffer_dict

    def _get_function_method_params_or_buffers(
        self, node: Node, traveled_nodes: Dict[Node, None], memo: Set[Node]
    ):
        parameter_dict = {}
        buffer_dict = {}
        for in_node in node.all_input_nodes:
            if self.is_get_attr_node(in_node):
                memo.add(in_node)
                traveled_nodes.setdefault(in_node)
                attr_module_name, _, attr_name = in_node.target.rpartition(".")
                attr_owner_module = self.graph_mod.get_submodule(attr_module_name)

                assert hasattr(attr_owner_module, attr_name), "Attribute not found."

                attr_v = getattr(attr_owner_module, attr_name)
                if isinstance(attr_v, torch.nn.Parameter):
                    parameter_dict[in_node.target] = attr_v
                elif attr_name in attr_owner_module._buffers:
                    buffer_dict[in_node.target] = (attr_v, attr_owner_module, attr_name)
        return parameter_dict, buffer_dict

    def remove_output(self, nodes: Dict[Node, None]):
        new_nodes: Dict[Node, None] = {}
        for node in nodes:
            if not self.is_output_node(node):
                new_nodes.setdefault(node)
        return new_nodes

    def remove_placeholder(self, nodes: Dict[Node, None]):
        new_nodes: Dict[Node, None] = {}
        for node in nodes:
            if not self.is_placeholder_node(node):
                new_nodes.setdefault(node)
        return new_nodes

    def is_params_and_buffer_all_collected(self):
        origin_all_params = dict(self.graph_mod.named_parameters())
        for k, v in origin_all_params.items():
            if k not in self.all_collected_params:
                print(f"Parameter {k} is not collected.")
                return False
        origin_all_buffers = dict(self.graph_mod.named_buffers())
        for k, v in origin_all_buffers.items():
            if k not in self.all_collected_buffers and not self.is_ignored_buffer(k):
                print(f"Buffer {k} is not collected.")
                return False
        return True

    def is_ignored_param(self, k: str):
        if hasattr(self, "ignored_params"):
            for ignored_pattern in self.ignored_params:
                if re.search(ignored_pattern, k):
                    return True
        return False

    def is_ignored_buffer(self, k: str):
        if hasattr(self, "ignored_buffers"):
            for ignored_pattern in self.ignored_buffers:
                if re.search(ignored_pattern, k):
                    return True
        return False

    def set_ignored_params(self, patern: List[str]):
        self.ignored_params = patern

    def set_ignore_buffers(self, patern: List[str]):
        self.ignored_buffers = patern

    def finalize(self) -> GraphModule:
        return super().finalize()


@register_pass("OnDemandMemoryPlan")
class OnDemandMemoryPlanPass(MemoryPlanPass):
    def __init__(self, m: Union[torch.nn.Module, GraphModule], max_depth=0):
        super().__init__(m, max_depth=max_depth)

    def run_on_graph(self) -> None:
        """TODO
        1. find the first part of parameters and buffers to be preloaded.
        2. insert preloading, guarding, unloading hook to enable the pipeline.
        """
        self.collect_params_and_buffers()
        self.process_plan()

    def collect_params_and_buffers(self):
        self.build_goal_classifiers(node_types=["output", "scatter"])
        self.set_ignore_buffers([r"\.load\_history", r"\.capacity\_history"])
        memo = set()
        placeholder_nodes = self.find_all_placeholders()
        (
            goal_nodes,
            traveled_nodes,
            parameters_dict,
            buffers_dict,
        ) = self.travel_to_goal(start_nodes=placeholder_nodes, memo=memo)
        self.add_new_plan_group(traveled_nodes, parameters_dict, buffers_dict)
        start_nodes = self.remove_output(goal_nodes)

        self.build_goal_classifiers(node_types=["output", "router"])

        while start_nodes:
            new_goal_nodes: Dict[Node, None] = {}
            for node in start_nodes:
                if self.is_scatter_node(node):
                    path_nodes = self.cluster_path_start_nodes(node)
                    for per_path_nodes in path_nodes:
                        (
                            goal_nodes,
                            traveled_nodes,
                            parameters_dict,
                            buffers_dict,
                        ) = self.travel_to_goal(start_nodes=per_path_nodes, memo=memo)
                        new_goal_nodes.update(goal_nodes)
                        self.add_new_plan_group(
                            traveled_nodes, parameters_dict, buffers_dict, node
                        )
                elif self.is_gather_node(node):
                    (
                        goal_nodes,
                        traveled_nodes,
                        parameters_dict,
                        buffers_dict,
                    ) = self.travel_to_goal(start_nodes=[node], memo=memo)
                    new_goal_nodes.update(goal_nodes)
                    self.add_new_plan_group(
                        traveled_nodes, parameters_dict, buffers_dict, node
                    )
            start_nodes = self.remove_output(new_goal_nodes)

        assert self.is_params_and_buffer_all_collected()

    def process_plan(self):
        orderred_plan_groups = self.sort_plan_groups()
        event_num = 0
        for plan_group in orderred_plan_groups.values():
            event_num += len(plan_group)
        MemoryPlanContext.init(event_num)
        event_id = 0
        for head_node, plan_group in orderred_plan_groups.items():
            for path_id, (
                tail_node,
                traveled_nodes,
                collected_params,
                collected_buffers,
            ) in enumerate(plan_group):
                memory_loader = OnDemandLoader(
                    path_id, event_id, collected_params, collected_buffers
                )
                memory_guarder = OnDemandGuarder(path_id, event_id)
                memory_unloader = OnDemandUnloader(
                    event_id, collected_params, collected_buffers
                )
                setattr(self.graph_mod, f"brt_memory_loader_{event_id}", memory_loader)
                setattr(
                    self.graph_mod, f"brt_memory_guarder_{event_id}", memory_guarder
                )
                setattr(
                    self.graph_mod, f"brt_memory_unloader_{event_id}", memory_unloader
                )
                event_id += 1

        for head_node, plan_group in reversed(orderred_plan_groups.items()):
            for path_id in range(len(plan_group)):
                tail_node = plan_group[-path_id - 1][0]
                with self.graph_mod.graph.inserting_after(tail_node):
                    unloader_node = self.graph_mod.graph.call_module(
                        f"brt_memory_unloader_{event_id - path_id -1}",
                        args=(tail_node,),
                    )
                tail_node.replace_all_uses_with(
                    unloader_node, lambda usr: usr != unloader_node
                )

            for path_id in range(len(plan_group)):
                with self.graph_mod.graph.inserting_after(head_node):
                    guarder_node = self.graph_mod.graph.call_module(
                        f"brt_memory_guarder_{event_id - path_id - 1}",
                        args=(head_node,),
                    )
                    head_node.replace_all_uses_with(
                        guarder_node, lambda usr: usr != guarder_node
                    )

            for path_id in range(len(plan_group)):
                with self.graph_mod.graph.inserting_after(head_node):
                    loader_node = self.graph_mod.graph.call_module(
                        f"brt_memory_loader_{event_id - path_id - 1}", args=(head_node,)
                    )
                    head_node.replace_all_uses_with(
                        loader_node, lambda usr: usr != loader_node
                    )

    def finalize(self) -> GraphModule:
        return super().finalize()


@register_pass("PredictMemoryPlan")
class PredictMemoryPlanPass(OnDemandMemoryPlanPass):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        lazy_load_threshold: float = 0.0,
        max_depth=0,
    ):
        super().__init__(m, max_depth=max_depth)
        self.lazy_load_threshold = lazy_load_threshold

    def run_on_graph(self) -> None:
        self.collect_params_and_buffers()
        self.process_plan()

    def process_plan(self):
        orderred_plan_groups = self.sort_plan_groups()
        event_num = 0
        for plan_group in orderred_plan_groups.values():
            event_num += len(plan_group)
        MemoryPlanContext.init(event_num)
        prev_head_node: Node = None
        event_id = 0
        for head_node, plan_group in orderred_plan_groups.items():
            for path_id, (
                tail_node,
                traveled_nodes,
                collected_params,
                collected_buffers,
            ) in enumerate(plan_group):
                if self.is_router_node(head_node):
                    head_node_m = self.sub_modules[head_node.target]
                    if self.is_scatter_node(head_node):
                        path_load_history = head_node_m.load_history[path_id].item()
                    elif self.is_gather_node(head_node):
                        path_load_history = head_node_m.load_history.mean().item()
                    else:
                        raise RuntimeError(
                            f"Unknown router kind: {head_node_m.__class__}"
                        )
                    if path_load_history >= self.lazy_load_threshold:
                        memory_loader = PredictLoader(
                            event_id, collected_params, collected_buffers
                        )
                        memory_guarder = PredictGuarder(event_id)
                        memory_unloader = PredictUnloader(
                            event_id, collected_params, collected_buffers
                        )
                    else:
                        memory_loader = OnDemandLoader(
                            path_id, event_id, collected_params, collected_buffers
                        )
                        memory_guarder = OnDemandGuarder(path_id, event_id)
                        memory_unloader = OnDemandUnloader(
                            event_id, collected_params, collected_buffers
                        )
                else:
                    memory_loader = PredictLoader(
                        event_id, collected_params, collected_buffers
                    )
                    memory_guarder = PredictGuarder(event_id)
                    memory_unloader = PredictUnloader(
                        event_id, collected_params, collected_buffers
                    )
                setattr(self.graph_mod, f"brt_memory_loader_{event_id}", memory_loader)
                setattr(
                    self.graph_mod, f"brt_memory_guarder_{event_id}", memory_guarder
                )
                setattr(
                    self.graph_mod, f"brt_memory_unloader_{event_id}", memory_unloader
                )
                event_id += 1

        for head_node, plan_group in orderred_plan_groups.items():
            for path_id in range(len(plan_group)):
                tail_node = plan_group[-path_id - 1][0]
                with self.graph_mod.graph.inserting_after(tail_node):
                    unloader_node = self.graph_mod.graph.call_module(
                        f"brt_memory_unloader_{event_id - path_id -1}",
                        args=(tail_node,),
                    )
                tail_node.replace_all_uses_with(
                    unloader_node, lambda usr: usr != unloader_node
                )

            for path_id in range(len(plan_group)):
                with self.graph_mod.graph.inserting_after(head_node):
                    guarder_node = self.graph_mod.graph.call_module(
                        f"brt_memory_guarder_{event_id - path_id - 1}",
                        args=(head_node,),
                    )
                    head_node.replace_all_uses_with(
                        guarder_node, lambda usr: usr != guarder_node
                    )

            for path_id in range(len(plan_group)):
                memory_loader = getattr(
                    self.graph_mod, f"brt_memory_loader_{event_id - path_id - 1}"
                )
                if isinstance(memory_loader, OnDemandLoader) or prev_head_node is None:
                    with self.graph_mod.graph.inserting_after(head_node):
                        loader_node = self.graph_mod.graph.call_module(
                            f"brt_memory_loader_{event_id - path_id - 1}",
                            args=(head_node,),
                        )
                        head_node.replace_all_uses_with(
                            loader_node, lambda usr: usr != loader_node
                        )
                elif isinstance(memory_loader, PredictLoader):
                    with self.graph_mod.graph.inserting_after(prev_head_node):
                        loader_node = self.graph_mod.graph.call_module(
                            f"brt_memory_loader_{event_id - path_id - 1}",
                            args=(prev_head_node,),
                        )
                        prev_head_node.replace_all_uses_with(
                            loader_node, lambda usr: usr != loader_node
                        )
                else:
                    raise RuntimeError(
                        f"Unknown loader kind: {memory_loader.__class__}"
                    )
            prev_head_node = head_node

    def finalize(self) -> GraphModule:
        return super().finalize()
