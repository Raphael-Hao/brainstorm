# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union
import torch
import torch.nn as nn
from brt.passes.base import PassBase, register_pass
from brt.passes.utils import is_scatter, is_gather
from brt.router import RouterBase, ScatterRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.trace.initialize import get_init_arguments


@register_pass("dead_path_eliminate")
class DeadPathEliminatePass(PassBase):
    def __init__(self, m: nn.Module, runtime_load=0):
        super().__init__(m)
        self.runtime_load = runtime_load

    def run_on_graph(self):
        sub_modules = dict(self.graph_mod.named_modules())
        # eliminate dead path
        for node in self.graph_mod.graph.nodes:
            if node.op == "call_module":
                node_m = sub_modules[node.target]
                if is_gather(node_m):
                    dead_paths = []
                    load_histroy = sub_modules[node.target].load_history
                    for path_id, path_load in enumerate(load_histroy):
                        if path_load == 0:
                            dead_paths.append(path_id)
                    live_paths = [
                        path_id
                        for path_id in range(len(load_histroy))
                        if path_id not in dead_paths
                    ]
                    new_args = ([node.args[0][path_id] for path_id in live_paths],)
                    new_load_history = torch.tensor(
                        [load_histroy[path_id] for path_id in live_paths],
                        dtype=torch.float64,
                        device="cpu",
                    )
                    sub_modules[node.target].load_history = new_load_history
                    node.args = new_args

    def finalize(self):
        # modify the dummy gather router with placeholder combine fabric
        super().finalize()

        sub_modules = dict(self.graph_mod.named_modules())
        for node in self.graph_mod.graph.nodes:
            if node.op == "call_module":
                node_m = sub_modules[node.target]
                if is_gather(node_m):
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
    def __init__(self, m: nn.Module, permanent_load):
        super().__init__(m)
        self.permanent_load = permanent_load

    def run_on_graph(self):
        sub_modules = dict(self.graph_mod.named_modules())
        for node in self.graph_mod.graph.nodes:
            if node.op == "call_module":
                node_m = sub_modules[node.target]
                if is_scatter(node_m):
                    permanent_paths = []
                    load_histroy = node_m.load_history
                    for path_id, path_load in enumerate(load_histroy):
                        if path_load == self.permanent_load or path_load == 0:
                            permanent_paths.append(path_id)
                    if len(permanent_paths) == len(load_histroy) and len(permanent_paths) > 0:
                        if isinstance(node_m, ScatterRouter):
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
                elif is_gather(node_m):
                    permanent_paths = []
                    load_histroy = node_m.load_history
                    for path_id, path_load in enumerate(load_histroy):
                        if path_load == self.permanent_load:
                            permanent_paths.append(path_id)
                    if len(permanent_paths) == len(load_histroy) and len(permanent_paths) > 0:
                        node_m.fabric = make_fabric(
                            "identity_combine", node_m.fabric_kwargs
                        )

    def finalize(self):
        return super().finalize()
