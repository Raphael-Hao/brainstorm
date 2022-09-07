# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Type, Union, List, Dict
import torch
from torch.fx import GraphModule, Node
from brt.router import is_router
from brt.runtime import Registry
from brt.trace.graph import symbolic_trace


class PassBase:
    def __init__(self, m: Union[torch.nn.Module, GraphModule]):
        if isinstance(m, GraphModule):
            m.graph.lint()
            m.recompile()
            self.graph_mod = m
        else:
            self.graph_mod = symbolic_trace(m)
        self.sub_modules = dict(self.graph_mod.named_modules())

    def analyze(flow: torch.Tensor) -> None:
        raise NotImplementedError

    def run_on_graph(self) -> None:
        raise NotImplementedError

    def finalize(self) -> GraphModule:
        self.graph_mod.graph.eliminate_dead_code()
        self.graph_mod.recompile()
        return self.graph_mod

    def is_placehoder_node(self, n: Node):
        return n.op == "placeholder"

    def is_output_node(self, n: Node):
        return n.op == "output"

    def is_module_node(self, n: Node):
        return n.op == "call_module"

    def is_router_node(self, n: Node):
        if n.op == "call_module":
            m = self.sub_modules[n.target]
            if is_router(m):
                return True
        return False

    def is_scatter_node(self, n: Node):
        if n.op == "call_module":
            m = self.sub_modules[n.target]
            if is_router(m) and "scatter" in m._router_type:
                return True
        return False

    def is_gather_node(self, n: Node):
        if n.op == "call_module":
            m = self.sub_modules[n.target]
            if is_router(m) and "gather" in m._router_type:
                return True
        return False

    def find_all_placeholders(self):
        placeholder_nodes: Dict[Node, None] = {}
        for node in self.graph_mod.graph.nodes:
            if node.op == "placeholder":
                placeholder_nodes.setdefault(node)
        return placeholder_nodes

    def cluster_path_start_nodes(self, scatter_node: Node):
        scatter_m = self.sub_modules[scatter_node.target]
        assert self.is_gather_node(scatter_node), (
            f"Node {scatter_node} is not a scatter node "
            "for clustering the start nodes of specific path."
        )
        flow_num = scatter_m.fabric.flow_num
        path_num = scatter_m.load_history.numel()
        start_nodes: List[Dict[Node, None]] = [{} for _ in range(path_num)]

        if flow_num == 1:
            for node in scatter_node.users:
                # NOTE current we check if the node is a call_function of getitem through its name
                assert "getitem" in str(
                    node.target
                ), "The node is not a call_function of getitem"
                item_id = int(node.args[1])
                path_id = item_id if item_id >= 0 else item_id + path_num
                start_nodes[path_id].setdefault(node)
        else:
            for flow_node in scatter_node.users:
                # NOTE current we check if the node is a call_function of getitem through its name
                assert "getitem" in str(
                    flow_node.target
                ), "The node is not a call_function of getitem"
                for node in flow_node.users:
                    assert "getitem" in str(
                        node.target
                    ), "The node is not a call_function of getitem"
                    item_id = int(node.args[1])
                    path_id = item_id if item_id >= 0 else item_id + path_num
                    start_nodes[path_id].setdefault(node)
        return start_nodes

    def find_first_and_last_node(self, nodes: Dict[Node, None]):
        first_node = None
        last_node = None
        for node in self.graph_mod.graph.nodes:
            if node in nodes:
                first_node = node
                break
        for node in reversed(self.graph_mod.graph.nodes):
            if node in nodes:
                last_node = node
        return first_node, last_node


def register_pass(pass_class: type) -> None:
    return Registry.register_sub_cls(pass_class, PassBase)


def get_pass(pass_name: str) -> Type[PassBase]:
    return Registry.get_sub_cls(pass_name, PassBase)
