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


@register_pass("dead_path_eliminate")
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
        node_id = 0
        for node in self.graph_mod.graph.nodes:
            # if self.is_scatter_node(node):
            #     node_id += 1
            #     node_m = self.sub_modules[node.target]
            #     dead_paths = []
            #     load_histroy = node_m.load_history
            #     print('scatter node id', node_id)
            #     for path_id, path_load in enumerate(load_histroy):
            #         if path_load < 500  and path_load > 0:
            #             print( path_load)

            if self.is_gather_node(node):
                node_id += 1
                node_m = self.sub_modules[node.target]
                dead_paths = []
                load_histroy = node_m.load_history
                # print(' gather node id', node_id)
                for path_id, path_load in enumerate(load_histroy):
                    # if path_load < 500  and path_load > 0:
                    #     print( path_load)

                    if path_load <= self.dead_load:
                        dead_paths.append(path_id)
                live_paths = [
                    path_id
                    for path_id in range(len(load_histroy))
                    if path_id not in dead_paths
                ]
                # if len(dead_paths)!=0:
                #     print(f"Eliminate dead path {dead_paths} in {node}")
                #     print(f"Live path {live_paths} in {node}")
                #     print("len live", len(live_paths))
                new_args = ([node.args[0][path_id] for path_id in live_paths],)
                new_load_history = np.array(
                    [load_histroy[path_id] for path_id in live_paths],
                    dtype=np.float64,
                )
                node_m.load_history = new_load_history
                node.args = new_args

    def finalize(self):
        super().finalize()

        # modify the dummy gather router with placeholder combine fabric
        for node in self.graph_mod.graph.nodes:
            if self.is_gather_node(node):
                node_m = self.sub_modules[node.target]
                if node.args[0] == []:
                    assert (
                        self.runtime_load > 0
                    ), "runtime_load must be positive when placeholder combine fabric is needed"
                    new_kwargs = {
                        "sparse": True,
                        "runtime_load": self.runtime_load,
                        "ptu_grains": node_m.ptu_grain_history,
                        "ptu_dtypes": node_m.ptu_dtype_history,
                        "ptu_devices": node_m.ptu_device_history,
                    }
                    node_m.fabric_kwargs.update(new_kwargs)
                    node_m.fabric = make_fabric(
                        "placeholder_combine", node_m.fabric_kwargs
                    )

        return super().finalize()


@register_pass("permanent_path_fold")
class PermanentPathFoldPass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        upper_perm_load: float,
        lower_perm_load=0.0,
    ):
        super().__init__(m)
        self.upper_perm_load = upper_perm_load
        self.lower_perm_load = lower_perm_load

    def run_on_graph(self):
        for node in self.graph_mod.graph.nodes:
            if self.is_router_node(node):
                node_m = self.sub_modules[node.target]
                if self.is_scatter_node(node):
                    permanent_paths = []
                    load_histroy = node_m.load_history
                    for path_id, path_load in enumerate(load_histroy):
                        if (
                            path_load >= self.upper_perm_load
                            or path_load <= self.lower_perm_load
                        ):
                            permanent_paths.append(path_id)
                    if (
                        len(permanent_paths) == len(load_histroy)
                        and len(permanent_paths) > 0
                    ):
                        node_m.protocol = make_protocol(
                            "identity", node_m.protocol_kwargs
                        )
                        new_fabric_kwargs = {
                            "path_num": len(permanent_paths),
                        }
                        node_m.fabric_kwargs.update(new_fabric_kwargs)
                        node_m.fabric = make_fabric(
                            "identity_dispatch", node_m.fabric_kwargs
                        )
                elif self.is_gather_node(node):
                    permanent_paths = []
                    load_histroy = node_m.load_history
                    for path_id, path_load in enumerate(load_histroy):
                        if path_load >= self.upper_perm_load:
                            permanent_paths.append(path_id)
                    if (
                        len(permanent_paths) == len(load_histroy)
                        and len(permanent_paths) > 0
                    ):
                        # print('node target', node.target)
                        # print('load history', load_histroy)
                        node_m.fabric = make_fabric(
                            "identity_combine", node_m.fabric_kwargs
                        )

    def finalize(self):
        return super().finalize()
