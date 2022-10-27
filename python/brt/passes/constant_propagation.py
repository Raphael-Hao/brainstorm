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

    ## maybe specific on msdnet currently
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
                        i = 0
                        new_args_tuple = []
                        for key in node.users.copy().keys():
                            for keyarg in key.args:
                                if isinstance(keyarg, int):
                                    i = keyarg
                            for new_key in key.users.copy().keys():
                                for new_key1 in new_key.users.copy().keys():
                                    if not isinstance(new_key1.args[0], list):
                                        new_args = (node.args[0][i],)
                                    else:
                                        construct_new_args = []
                                        for tmp_node in new_key1.args[0]:
                                            if not tmp_node == new_key:
                                                construct_new_args.append(tmp_node)
                                            else:
                                                construct_new_args.append(
                                                    node.args[0][i]
                                                )
                                        new_args = (construct_new_args,)
                                    new_key1.args = new_args
                                self.graph_mod.graph.erase_node(new_key)

                            i += 1
                            self.graph_mod.graph.erase_node(key)
                        self.graph_mod.graph.erase_node(node)
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
                        i = 0
                        new_args_tuple = []
                        for key in node.users.copy().keys():
                            for keyarg in key.args:
                                if isinstance(keyarg, int):
                                    i = keyarg
                            if not isinstance(key.args[0], list):
                                new_args = (node.args[0][i],)
                            else:
                                construct_new_args = []
                                for tmp_node in key.args[0]:
                                    if not tmp_node == new_key:
                                        construct_new_args.append(tmp_node)
                                    else:
                                        construct_new_args.append(node.args[0][i])
                                new_args = (construct_new_args,)
                            key.args = new_args
                            i += 1
                        self.graph_mod.graph.erase_node(node)

    def finalize(self):
        self.graph_mod.graph.lint()
        return super().finalize()
