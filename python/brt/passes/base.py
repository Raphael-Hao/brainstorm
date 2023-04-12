# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Type, Union, List, Dict, Any

import operator
import torch
from torch.fx import GraphModule, Node
from brt.router import is_router
from brt.runtime import Registry, log
from brt.trace.graph import symbolic_trace

__all__ = ["PassBase", "register_pass", "get_pass"]

logger = log.get_logger(__file__)


class PassBase:
    def __init__(self, m: Union[torch.nn.Module, GraphModule]):
        if isinstance(m, GraphModule):
            m.graph.lint()
            m.recompile()
            self.graph_mod = m
        else:
            self.graph_mod = symbolic_trace(m)
        self.sub_modules = dict(self.graph_mod.named_modules())

    def __call__(self) -> GraphModule:
        self.run_on_graph()
        return self.finalize()

    def run_on_graph(self) -> None:
        raise NotImplementedError(
            "run_on_graph should be implemented by an actual Pass."
        )

    def finalize(self) -> GraphModule:
        self.graph_mod.graph.lint()

        self.graph_mod.graph.eliminate_dead_code()
        self.graph_mod.recompile()
        return self.graph_mod

    # common help function for node type check
    def is_placeholder_node(self, n: Node):
        return n.op == "placeholder"

    def is_get_attr_node(self, n: Node):
        return n.op == "get_attr"

    def is_function_node(self, n: Node):
        return n.op == "call_function"

    def is_module_node(self, n: Node):
        return n.op == "call_module"

    def is_method_node(self, n: Node):
        return n.op == "call_method"

    def is_output_node(self, n: Node):
        return n.op == "output"

    # brt-related help function for node type check
    def is_router_node(self, n: Node):
        if self.is_module_node(n):
            m = self.sub_modules[n.target]
            if is_router(m):
                return True
        return False

    def is_scatter_node(self, n: Node):
        if self.is_module_node(n):
            m = self.sub_modules[n.target]
            if is_router(m) and "scatter" in m._router_type:
                return True
        return False

    def is_gather_node(self, n: Node):
        if self.is_module_node(n):
            m = self.sub_modules[n.target]
            if is_router(m) and "gather" in m._router_type:
                return True
        return False

    def find_all_placeholders(self):
        placeholder_nodes: Dict[Node, None] = {}
        for node in self.graph_mod.graph.nodes:
            if self.is_placeholder_node(node):
                placeholder_nodes.setdefault(node)
        return placeholder_nodes

    def cluster_path_start_nodes(self, scatter_node: Node):
        scatter_m = self.sub_modules[scatter_node.target]
        assert self.is_scatter_node(scatter_node), (
            f"Node {scatter_node} is not a scatter node "
            "for clustering the start nodes of specific path."
        )
        flow_num = scatter_m.fabric.flow_num
        path_num = scatter_m.fabric.load_history.size
        start_nodes: List[Dict[Node, None]] = [{} for _ in range(path_num)]

        if flow_num == 1:
            for node in scatter_node.users:
                # NOTE current we check if the node is a call_function of getitem through its name
                assert (
                    operator.getitem == node.target
                ), "The node is not a call_function of getitem"
                item_id = int(node.args[1])
                path_id = item_id if item_id >= 0 else item_id + path_num
                start_nodes[path_id].setdefault(node)
        else:
            for flow_node in scatter_node.users:
                # NOTE current we check if the node is a call_function of getitem through its name
                assert (
                    operator.getitem == flow_node.target
                ), f"The node {flow_node} is not a call_function of getitem"
                for node in flow_node.users:
                    assert (
                        operator.getitem == node.target
                    ), f"The node {node} is not a call_function of getitem"
                    item_id = int(node.args[1])
                    path_id = item_id if item_id >= 0 else item_id + path_num
                    start_nodes[path_id].setdefault(node)
        return start_nodes

    def find_first_node(self, nodes: Dict[Node, None]) -> Node:
        for node in self.graph_mod.graph.nodes:
            if node in nodes:
                return node
        raise RuntimeError("All nodes are not in the graph.")

    def find_last_node(self, nodes: Dict[Node, None]) -> Node:
        for node in reversed(self.graph_mod.graph.nodes):
            if node in nodes:
                return node
        raise RuntimeError("All nodes are not in the graph.")

    def find_first_and_last_node(self, nodes: Dict[Node, None]):
        return self.find_first_node(nodes), self.find_last_node(nodes)

    def topo_compare(self, n1: Node, n2: Node):
        """compare the topo order of two nodes in the graph
        Args:
            n1 (Node): first node
            n2 (Node): sencond node
        """
        first_node = self.find_first_node({n1: None, n2: None})
        if first_node == n1:
            return True
        return False

    def sort_nodes(self, nodes: Dict[Node, None]):
        """sort the nodes in the graph by their topo order
        Args:
            nodes (Dict[Node, None]): the nodes to be sorted
        """
        sorted_nodes: List[Node] = []

        while nodes:
            first_node = self.find_first_node(nodes)
            sorted_nodes.append(first_node)
            nodes.pop(first_node)
        return sorted_nodes


def register_pass(pass_class: type) -> None:
    return Registry.register_sub_cls(pass_class, PassBase)


def get_pass(pass_name: str) -> Type[PassBase]:
    return Registry.get_sub_cls(pass_name, PassBase)
