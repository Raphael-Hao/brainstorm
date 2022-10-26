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
from brt.runtime import log
from brt.runtime.tensor_group import TensorGroupManager, TensorGroup
from brt.runtime.memory_planner import *

PLGT = Tuple[
    List[Node],
    Dict[str, nn.Parameter],
    Dict[str, Tuple[nn.Module, str]],
]

logger = log.get_logger(__file__)

__all__ = ["OnDemandMemoryPlan", "PredictMemoryPlan"]


@register_pass("MemoryPlan")
class MemoryPlanPass(PassBase):
    def __init__(
        self, m: Union[torch.nn.Module, GraphModule], max_depth=0, is_grouping=False
    ):
        super().__init__(m)
        self.max_depth = max_depth
        self.is_grouping = is_grouping
        self.goal_classifiers: List[Callable] = []
        assert (
            self.max_depth == 0
        ), f"Max_depth is set to {self.max_depth}, greater than 0 is not supported yet."
        self.plan_groups: Dict[Node, List[PLGT]] = dict()
        self.all_collected_params: Dict[str, torch.Tensor] = dict()
        self.all_collected_buffers: Dict[
            str, Tuple[torch.Tensor, nn.Module, str]
        ] = dict()
        self.initial_loaders: List[nn.Module] = []

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
        self.set_ignore_buffers(
            [r"\.load\_history", r"\.capacity\_history", r"\.ptu\_path\_history"]
        )

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

    def inject_planners(
        self,
        mode: str,
        path_id: int,
        event_id: int,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, Tuple[nn.Module, str]],
    ):
        """Inject planners into the graph module so that the memory planners
        can be called like a normal nn.Module during inference.

        Args:
            mode (str): planer mode, can be "predict", "on_demand", "inital"
            path_id (int): path id for scatter router
            event_id (int): event id for recording cuda events
            params (Dict[str, nn.Parameter]): collected parameters
            buffers (Dict[str, Tuple[nn.Module, str]]): collected buffers

        Raises:
            ValueError: Unsupported planner mode
        """
        if self.is_grouping:
            tensor_group = _group_params_buffers(params, buffers)
            if mode == "on_demand":
                memory_loader = GroupedOnDemandLoader(path_id, event_id, tensor_group)
                memory_guarder = GroupedOnDemandGuarder(path_id, event_id)
                memory_unloader = GroupedOnDemandUnloader(event_id, tensor_group)
            elif mode == "predict":
                memory_loader = GroupedPredictLoader(event_id, tensor_group)
                memory_guarder = GroupedPredictGuarder(event_id)
                memory_unloader = GroupedPredictUnloader(event_id, tensor_group)
            elif mode == "initial":
                memory_loader = GroupedInitialLoader(event_id, tensor_group)
            else:
                raise ValueError(f"Mode {mode} is not supported.")
        else:
            if mode == "on_demand":
                memory_loader = OnDemandLoader(path_id, event_id, params, buffers)
                memory_guarder = OnDemandGuarder(path_id, event_id)
                memory_unloader = OnDemandUnloader(event_id, params, buffers)
            elif mode == "predict":
                memory_loader = PredictLoader(event_id, params, buffers)
                memory_guarder = PredictGuarder(event_id)
                memory_unloader = PredictUnloader(event_id, params, buffers)
            elif mode == "initial":
                memory_loader = InitialLoader(event_id, params, buffers)
            else:
                raise ValueError(f"Mode {mode} is not supported.")
        if mode == "initial":
            setattr(self.graph_mod, f"brt_initial_loader_{event_id}", memory_loader)
            self.initial_loaders.append(memory_loader)
        else:
            setattr(self.graph_mod, f"brt_memory_loader_{event_id}", memory_loader)
            setattr(self.graph_mod, f"brt_memory_guarder_{event_id}", memory_guarder)
            setattr(self.graph_mod, f"brt_memory_unloader_{event_id}", memory_unloader)

    def inspect_planner_mode(self, event_id: int) -> str:
        loader = getattr(self.graph_mod, f"brt_initial_loader_{event_id}", None)
        if loader is not None:
            return "initial"
        loader = getattr(self.graph_mod, f"brt_memory_loader_{event_id}", None)
        if isinstance(loader, (PredictLoader, GroupedPredictLoader)):
            return "predict"
        elif isinstance(loader, (OnDemandLoader, GroupedOnDemandLoader)):
            return "on_demand"
        else:
            raise ValueError(f"Unrecognized mode for planner with event id {event_id}")

    def add_new_plan_group(
        self,
        traveled_nodes: Dict[Node, None],
        collected_params: Dict[str, nn.Parameter],
        collected_buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]],
        head_node: Node = None,
        head_node_include=True,
    ):
        sorted_traveled_nodes = self.sort_nodes(traveled_nodes)
        first_node, last_node = sorted_traveled_nodes[0], sorted_traveled_nodes[-1]
        if head_node is None:
            if head_node_include:
                head_node = first_node
            else:
                head_node = first_node.prev

        self.plan_groups.setdefault(head_node, []).append(
            (sorted_traveled_nodes, collected_params, collected_buffers)
        )
        self.all_collected_params.update(collected_params)
        self.all_collected_buffers.update(collected_buffers)

    def sort_plan_groups(self):
        orderred_plan_groups: typing.OrderedDict[Node, List[PLGT]] = OrderedDict()
        for node in self.graph_mod.graph.nodes:
            if node in self.plan_groups:
                orderred_plan_groups[node] = self.plan_groups[node]
        return orderred_plan_groups

    def sort_plan_groups_in_tuple(self):
        orderred_plan_groups: List[Tuple[Node, List[PLGT]]] = list()
        for node in self.graph_mod.graph.nodes:
            if node in self.plan_groups:
                orderred_plan_groups.append((node, self.plan_groups[node]))
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
                logger.debug(f"Parameter {k} is not collected.")
                return False
        origin_all_buffers = dict(self.graph_mod.named_buffers())
        for k, v in origin_all_buffers.items():
            if k not in self.all_collected_buffers and not self.is_ignored_buffer(k):
                logger.debug(f"Buffer {k} is not collected.")
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
    def __init__(
        self, m: Union[torch.nn.Module, GraphModule], max_depth=0, is_grouping=False
    ):
        super().__init__(m, max_depth=max_depth, is_grouping=is_grouping)

    def run_on_graph(self) -> None:
        """TODO
        1. find the first part of parameters and buffers to be preloaded.
        2. insert preloading, guarding, unloading hook to enable the pipeline.
        """
        self.collect_params_and_buffers()
        self.process_plan()

    def collect_params_and_buffers(self):
        self.build_goal_classifiers(node_types=["output", "scatter"])
        self.set_ignore_buffers(
            [r"\.load\_history", r"\.capacity\_history", r"\.ptu\_decision\_history"]
        )
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
        orderred_plan_groups = self.sort_plan_groups_in_tuple()
        event_num = 0
        for _, plan_groups in orderred_plan_groups:
            event_num += len(plan_groups)

        MemoryPlanContext.init(event_num)
        event_id = 0
        initial_loader_injected = False
        for head_node, plan_groups in orderred_plan_groups:
            for path_id, (
                _traveled_nodes,
                collected_params,
                collected_buffers,
            ) in enumerate(plan_groups):
                if not initial_loader_injected:
                    mode = "initial"
                    initial_loader_injected = True
                else:
                    mode = "on_demand"
                self.inject_planners(
                    mode, path_id, event_id, collected_params, collected_buffers
                )
                event_id += 1

        event_id -= 1
        for head_node, plan_groups in reversed(orderred_plan_groups):
            for path_id in range(len(plan_groups)):
                planner_mode = self.inspect_planner_mode(event_id - path_id)
                if planner_mode == "initial":
                    continue
                traveled_nodes = plan_groups[-path_id - 1][0]
                tail_node = traveled_nodes[-1]
                with self.graph_mod.graph.inserting_after(tail_node):
                    unloader_node = self.graph_mod.graph.call_module(
                        f"brt_memory_unloader_{event_id - path_id}",
                        args=(tail_node,),
                    )
                    tail_node.replace_all_uses_with(
                        unloader_node, lambda usr: usr != unloader_node
                    )

            for path_id in range(len(plan_groups)):
                planner_mode = self.inspect_planner_mode(event_id - path_id)
                if planner_mode == "initial":
                    continue
                with self.graph_mod.graph.inserting_after(head_node):
                    guarder_node = self.graph_mod.graph.call_module(
                        f"brt_memory_guarder_{event_id - path_id}",
                        args=(head_node,),
                    )
                    head_node.replace_all_uses_with(
                        guarder_node, lambda usr: usr != guarder_node
                    )

            for path_id in range(len(plan_groups)):
                planner_mode = self.inspect_planner_mode(event_id - path_id)
                if planner_mode == "initial":
                    continue
                with self.graph_mod.graph.inserting_after(head_node):
                    loader_node = self.graph_mod.graph.call_module(
                        f"brt_memory_loader_{event_id - path_id}", args=(head_node,)
                    )
                    head_node.replace_all_uses_with(
                        loader_node, lambda usr: usr != loader_node
                    )
            event_id -= len(plan_groups)

    def finalize(self) -> Tuple[List[nn.Module], GraphModule]:
        return self.initial_loaders, super().finalize()


@register_pass("PredictMemoryPlan")
class PredictMemoryPlanPass(OnDemandMemoryPlanPass):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        lazy_load_threshold: float = 0.0,
        max_depth=0,
        is_grouping=False,
    ):
        super().__init__(m, max_depth=max_depth, is_grouping=is_grouping)
        self.lazy_load_threshold = lazy_load_threshold

    def run_on_graph(self) -> None:
        self.collect_params_and_buffers()
        self.process_plan()

    def process_plan(self):
        orderred_plan_groups = self.sort_plan_groups_in_tuple()
        event_num = 0
        for plan_groups in orderred_plan_groups:
            event_num += len(plan_groups)

        MemoryPlanContext.init(event_num)

        event_id = 0
        prev_head_node = None
        for head_node, plan_groups in orderred_plan_groups:
            for path_id, (
                traveled_nodes,
                collected_params,
                collected_buffers,
            ) in enumerate(plan_groups):
                if self.is_router_node(head_node):
                    head_node_m = self.sub_modules[head_node.target]
                    if self.is_scatter_node(head_node):
                        path_load_history = head_node_m.load_history[path_id]
                    elif self.is_gather_node(head_node):
                        path_load_history = head_node_m.load_history.max()
                    else:
                        raise RuntimeError(
                            f"Unknown router kind: {head_node_m.__class__}"
                        )
                    if path_load_history >= self.lazy_load_threshold:
                        if prev_head_node is None:
                            mode = "initial"
                        else:
                            mode = "predict"
                        self.inject_planners(
                            "predict",
                            path_id,
                            event_id,
                            collected_params,
                            collected_buffers,
                        )
                    else:
                        self.inject_planners(
                            "on_demand",
                            path_id,
                            event_id,
                            collected_params,
                            collected_buffers,
                        )
                else:
                    if prev_head_node is None:
                        mode = "initial"
                    else:
                        mode = "predict"
                    self.inject_planners(
                        mode,
                        path_id,
                        event_id,
                        collected_params,
                        collected_buffers,
                    )
                event_id += 1
            prev_head_node = head_node

        orderred_plan_groups.reverse()
        event_id -= 1
        self.initial_loaders: List[nn.Module] = []

        for g_id, (head_node, plan_groups) in enumerate(orderred_plan_groups):
            for path_id in range(len(plan_groups)):
                planner_mode = self.inspect_planner_mode(event_id - path_id)
                if planner_mode == "initial":
                    continue
                traveled_nodes = plan_groups[-path_id - 1][0]
                tail_node = traveled_nodes[-1]
                with self.graph_mod.graph.inserting_after(tail_node):
                    unloader_node = self.graph_mod.graph.call_module(
                        f"brt_memory_unloader_{event_id - path_id}", args=(tail_node,)
                    )
                    tail_node.replace_all_uses_with(
                        unloader_node, lambda usr: usr != unloader_node
                    )

            for path_id in range(len(plan_groups)):
                planner_mode = self.inspect_planner_mode(event_id - path_id)
                if planner_mode == "initial":
                    continue
                with self.graph_mod.graph.inserting_after(head_node):
                    guarder_node = self.graph_mod.graph.call_module(
                        f"brt_memory_guarder_{event_id - path_id}", args=(head_node,)
                    )
                    head_node.replace_all_uses_with(
                        guarder_node, lambda usr: usr != guarder_node
                    )

            for path_id in range(len(plan_groups)):
                planner_mode = self.inspect_planner_mode(event_id - path_id)
                if planner_mode == "initial":
                    continue
                memory_loader = getattr(
                    self.graph_mod, f"brt_memory_loader_{event_id - path_id}"
                )
                if isinstance(
                    memory_loader, (OnDemandLoader, GroupedOnDemandLoader)
                ) or (g_id + 1) == len(orderred_plan_groups):
                    with self.graph_mod.graph.inserting_after(head_node):
                        loader_node = self.graph_mod.graph.call_module(
                            f"brt_memory_loader_{event_id - path_id}", args=(head_node,)
                        )
                        head_node.replace_all_uses_with(
                            loader_node, lambda usr: usr != loader_node
                        )
                elif isinstance(memory_loader, (PredictLoader, GroupedPredictLoader)):
                    prev_head_node = orderred_plan_groups[g_id + 1][0]
                    with self.graph_mod.graph.inserting_after(prev_head_node):
                        loader_node = self.graph_mod.graph.call_module(
                            f"brt_memory_loader_{event_id - path_id}",
                            args=(prev_head_node,),
                        )
                        prev_head_node.replace_all_uses_with(
                            loader_node, lambda usr: usr != loader_node
                        )
                else:
                    raise RuntimeError(
                        f"Unknown loader kind: {memory_loader.__class__}"
                    )
            event_id -= len(plan_groups)


def _group_params_buffers(
    params: Dict[str, nn.Parameter],
    buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]],
    target_device=None,
) -> TensorGroup:
    """Group params and buffers into a single tensor.

    Args:
        params (Dict[str, nn.Parameter]): collected params
        buffers (Dict[str, nn.Parameter]): collected buffers

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: single pin memory and cuda tensor containing all params and buffers
    """
    total_size_in_bytes = 0

    for p in params.values():
        if p is not None:
            total_size_in_bytes += p.data.numel() * p.data.element_size()
        if p.grad is not None:
            total_size_in_bytes += p.grad.data.numel() * p.grad.data.element_size()
    for b_t, b_mod, b_name in buffers.values():
        if b_mod._buffers[b_name] is not None:
            total_size_in_bytes += b_t.numel() * b_t.element_size()

    tenosr_group = TensorGroupManager.acquire_tensor_group(
        total_size_in_bytes, target_device
    )

    for p in params.values():
        if p is None:
            continue
        tenosr_group.include_tensor(p)
        if p.grad is not None:
            tenosr_group.include_tensor(p.grad)

    for b_t, b_mod, b_name in buffers.values():
        if b_mod._buffers[b_name] is None:
            continue
        tenosr_group.include_tensor(b_t)

    return tenosr_group
