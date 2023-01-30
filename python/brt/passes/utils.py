# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union, Dict, Any

from torch.fx import Node

from brt.trace.graph import Graph, Node

__all__ = ["debug_node"]


def debug_node(node: Node):
    print(
        f"node:{node}, op:{node.op}, target:{node.target}, users:{node.users}, args:{node.args}, input_nodes:{node.all_input_nodes}"
    )

def build_sub_graph(
    graph: Graph, nodes: List[Node], outputs: Union[List[Node], Node, int, None] = -1
) -> Graph:
    """Build a subgraph with `nodes` from `graph`, also generate inputs and outputs.

    Args:
        graph (Graph): The graph building from.
        nodes (List[Node]): The node of new subgraph.
        outputs (Union[List[Node], Node, int, None], optional): The return value node. Defaults to -1.
            None: Return None.
            int: Return `nodes[outputs]`.
            Node | List[Node]: Return `outputs`.

    Returns:
        Graph: the subgraph
    """
    new_graph = Graph()
    node_remap = {}
    # insert nodes
    for node in nodes:
        assert len(node.kwargs) == 0, node.kwargs
        for arg in node.args:
            if isinstance(arg, Node) and arg not in node_remap:
                # node_remap[arg] = new_graph.placeholder(arg.name, arg.type)
                node_remap[arg] = new_graph.create_node(
                    "placeholder",
                    arg.name,
                    name=arg.name,
                    type_expr=arg.type,
                    is_fixed_inout=arg.is_fixed_inout,
                    inshape=arg.inshape,
                    outshape=arg.outshape,
                )
        node_remap[node] = new_graph.node_copy(node, lambda n: node_remap[n])
    # add output
    if outputs is not None:
        if isinstance(outputs, int):
            outputs = nodes[outputs]
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
        outputs = tuple(node_remap[o] for o in outputs)
        new_graph.create_node(
            op="output",
            target="output",
            args=outputs,
            name="output",
        )
    return new_graph