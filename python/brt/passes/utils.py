# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from brt.router import RouterBase
from torch.fx import Node

__all__ = ["is_scatter", "is_gather"]


def is_scatter(m):
    if isinstance(m, RouterBase) and "scatter" in m._router_type:
        return True
    return False


def is_gather(m):
    if isinstance(m, RouterBase) and "gather" in m._router_type:
        return True
    return False


def debug_node(node: Node):
    print(
        f"{node} | {node.op} | {node.target} | {node.users} | {node.args} | {node.all_input_nodes}"
    )
