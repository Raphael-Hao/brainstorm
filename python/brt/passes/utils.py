# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from torch.fx import Node

__all__ = ["debug_node"]


def debug_node(node: Node):
    print(
        f"node:{node}, op:{node.op}, target:{node.target}, users:{node.users}, args:{node.args}, input_nodes:{node.all_input_nodes}"
    )
