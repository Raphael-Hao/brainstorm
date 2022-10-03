# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union

import torch
from torch.fx import GraphModule
import numpy as np
from brt.passes.base import PassBase, register_pass
from brt.router import ScatterRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol


@register_pass("op_transform")
class DeadPathEliminatePass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        dead_load=0.0,
        runtime_load: int = 0,
    ):
        super().__init__(m)
        self.dead_load = dead_load
        self.runtime_load = runtime_load

    def run_on_graph(self):
        # eliminate dead path
        for node in self.graph_mod.graph.nodes:
            if self.is_gather_node(node):
                node_m = self.sub_modules[node.target]
                dead_paths = []
                load_histroy = node_m.load_history
                for path_id, path_load in enumerate(load_histroy):
                    if path_load <= self.dead_load:
                        dead_paths.append(path_id)
                live_paths = [
                    path_id
                    for path_id in range(len(load_histroy))
                    if path_id not in dead_paths
                ]
                new_args = ([node.args[0][path_id] for path_id in live_paths],)
                new_load_history = np.array(
                    [load_histroy[path_id] for path_id in live_paths],
                    dtype=np.float64,
                )
                node_m.load_history = new_load_history
                node.args = new_args

    def finalize(self):
        return super().finalize()
