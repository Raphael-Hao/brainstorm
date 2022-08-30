# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from brt.passes.base import PassBase, register_pass
from brt.router import GatherRouter, ScatterRouter


@register_pass("dead_path_eliminate")
class DeadPathEliminatePass(PassBase):
    def __init__(self, m: nn.Module):
        super().__init__(m)

    def run_on_graph(self):
        sub_modules = dict(self.graph_mod.named_modules())
        for node in self.graph_mod.graph.nodes:
            if node.op == "call_module":
                if isinstance(sub_modules[node.target], GatherRouter):
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
        return super().finalize()


@register_pass("permanent_path_fold")
class PermanentPathFoldPass(PassBase):
    def __init__(self, permanent_target, m: nn.Module):
        super().__init__(m)
        self.perm_target = permanent_target

    def run_on_graph(self):
        sub_modules = self.graph_mod.named_modules()
        for node in self.graph.nodes:
            if node.op == "call_module":
                if isinstance(sub_modules[node.target], ScatterRouter):
                    permanent_paths = []
                    load_histroy = sub_modules[node.target].load_history
                    for path_id, path_load in enumerate(load_histroy):
                        if path_load == 0:
                            permanent_paths.append(path_id)
                    if len(permanent_paths) == len(load_histroy):
                        traced_kwargs = {}
                elif isinstance(sub_modules[node.target], GatherRouter):
                    permanent_paths = []
                    load_histroy = sub_modules[node.target].load_history
                    for path_id, path_load in enumerate(load_histroy):
                        if path_load == 0:
                            permanent_paths.append(path_id)
                    if len(permanent_paths) == len(load_histroy):
                        traced_kwargs = {}

    def finalize(self):
        return super().finalize()
