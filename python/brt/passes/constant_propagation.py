# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union

import operator

import numpy as np

import torch
from torch.fx import GraphModule

from brt.runtime.log import get_logger
from brt.passes.base import PassBase, register_pass
from brt.router import ScatterRouter, GatherRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.trace import Node

logger = get_logger("__file__")


@register_pass("constant_propagation")
class ConstantPropagationPass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        upper_perm_load: float,
        lower_perm_load=0.0,
    ):
        super().__init__(m)
        self.lower_perm_load = lower_perm_load
        self.upper_perm_load = upper_perm_load

    def run_on_graph(self):
        graph = self.graph_mod.graph
        for node in graph.nodes:
            if self.is_scatter_node(node):
                scatter_node: Node = node
                scatter: ScatterRouter = self.sub_modules[scatter_node.target]
                # Notation: flow_0, flow_1 -> scatter -> [outflow_0_path_0, outflow_0_path_1], [outflow_1_path_0, outflow_1_path_1]
                flow_num = scatter.fabric.flow_num
                load_history = scatter.load_history
                path_num = len(load_history)
                ptu_decision_history = scatter.ptu_decision_history
                # Generate permanent path indecies
                is_empty_path = [False] * path_num
                is_forwarding_path = [False] * path_num
                for i, path_history in enumerate(ptu_decision_history):
                    if path_history.size == 0:
                        is_empty_path[i] = True
                    else:
                        # path_history is like a range(*)
                        if (np.diff(path_history, 1) == 1).all():
                            is_forwarding_path[i] = True

                # Check if all path is permanent
                # TODO: If there is a dynamic path but not used (e.g. no getitem), the scatter is still fixed
                if not all(
                    is_empty ^ is_forwarding
                    for is_empty, is_forwarding in zip(is_empty_path, is_forwarding_path)
                ):
                    continue
                # Edit the path
                if flow_num == 1:
                    logger.debug(
                        f"Currently not support constant propagation for scatters with flow_num = 1"
                    )
                    continue
                for flow_user in list(scatter_node.users.keys()):
                    assert (
                        flow_user.op == "call_function"
                        and flow_user.target == operator.getitem
                    ), f"Unknown flow users, {flow_user.op=}, {flow_user.target}"
                    flow_index = flow_user.args[1]
                    for path_user in list(flow_user.users.keys()):
                        assert (
                            path_user.op == "call_function"
                            and path_user.target == operator.getitem
                        ), f"Unknown path users in flow {flow_index}, {path_user.op=}, {path_user.target}"
                        path_index = path_user.args[1]
                        if is_empty_path[path_index]:
                            # TODO: delete path
                            try:
                                graph.erase_node(path_user)
                            except:
                                logger.debug(
                                    f"Cannot erase path user, scatter: {scatter_node.name}, flow: {flow_index}, path: {path_index}"
                                )
                        else:  # is_forwarding_path[path_index]
                            path_user.replace_all_uses_with(scatter_node.args[0][flow_index])
                            graph.erase_node(path_user)
                    try:
                        graph.erase_node(flow_user)
                    except:
                        logger.debug(
                            f"Cannot erase flow user, scatter: {scatter_node.name}, flow: {flow_index}"
                        )
                try:
                    graph.erase_node(scatter_node)
                except:
                    logger.debug(
                        f"Cannot erase constant scatter node, scatter: {scatter_node.name}"
                    )
            elif self.is_gather_node(node):
                gather_node: Node = node
                gather: GatherRouter = self.sub_modules[gather_node.target]
                flow_num = gather.fabric.flow_num
                if flow_num != 1:
                    logger.debug(f"Currently not support constant propagation for gathers with flow_num != 1, get {flow_num}")
                load_history = gather.load_history
                path_num = len(load_history)
                if np.count_nonzero(load_history) == 0:
                    # TODO: Delete gather
                    raise NotImplementedError
                elif np.count_nonzero(load_history) == 1:
                    # Straight forward
                    path_index = int(load_history.nonzero()[0])
                    gather_node.replace_all_uses_with(gather_node.args[0][path_index])
                    graph.erase_node(gather_node)


    def finalize(self):
        self.graph_mod.graph.lint()
        return super().finalize()
