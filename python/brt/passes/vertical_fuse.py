# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union, Dict, Any, Set

from copy import deepcopy
from hashlib import new
import itertools as itt
import operator

import torch
import torch.nn as nn
from torch import fx
from torch.fx.immutable_collections import immutable_list, immutable_dict
from torch.fx.node import map_arg

from brt.runtime import log
from brt.runtime.grid_tensor import (
    init_grid_tensor,
    init_grid_tensor_from,
    deinit_grid_tensor,
)
from brt.jit import make_jit_module
from brt.jit.modules.factory import JitModuleFactory
from brt.router import ScatterRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.trace.graph import symbolic_trace, GraphModule, Graph, Node

# import brt.passes.fuse_util
from brt.passes.base import PassBase, register_pass
from brt.passes.utils import *

logger = log.get_logger(__file__)


@register_pass("vertical_fuse")
class VerticalFusePass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        sample_inputs: Dict[str, Any],
        fusing_head: bool = False,
        **tracer_kwargs,
    ):
        override_tracer_kwargs = {"fixed_inputs": bool(fusing_head)}
        tracer_kwargs.update(override_tracer_kwargs)
        self.sub_modules = dict(m.named_modules())
        if isinstance(m, GraphModule) and m.graph.is_shape_tracked:
            m.graph.lint()
            m.recompile()
            self.graph_mod = m
        else:
            self.graph_mod = symbolic_trace(
                m, tracing_shape=True, sample_inputs=sample_inputs, **tracer_kwargs
            )
        self.origin_graph = self.graph_mod.graph
        self.fusing_head = fusing_head

    def find_fuse_parteners(
        self,
        start: Node,
        visited: List[Node] = None,
    ) -> Union[List[Node], None]:
        if visited is None:
            visited = lambda n: True

        if not (self.is_module_node(start) and start.is_fixed_inout):
            logger.debug(f"start node `{start.name}` should be a fixed module node")
            return None

        cur_node: Node = start
        fusing_nodes = []
        is_last_try = False
        while not is_last_try:
            fusing_nodes.append(cur_node)
            if cur_node in visited or not is_at_wavefront(
                cur_node, visited | set(fusing_nodes)
            ):
                logger.debug(f"node `%{cur_node.name}` is not aviliable")
                break
            if self.is_router_node(cur_node):
                logger.debug(f"node `%{cur_node.name}` is a router node")
                break
            if not (
                self.is_module_node(cur_node)
                or self.is_method_node(cur_node)
                or self.is_function_node(cur_node)
            ):
                logger.debug(f"node `%{cur_node.name}` is not callable")
                break
            if not cur_node.is_fixed_inout:
                logger.debug(f"node `%{cur_node.name}` is not fixed")
                break
            if not isinstance(cur_node.outshape, torch.Size):
                logger.debug(f"node `%{cur_node.name}` has more than 1 outputs")
                break
            if len(cur_node.kwargs) != 0:
                logger.debug(f"node `%{cur_node.name}` has kwargs {cur_node.kwargs}")
                break
            if len(cur_node.users) > 1:
                logger.debug(f"node `%{cur_node.name}` has more than 1 users, last try")
                is_last_try = True
            try:
                for arg in cur_node.args:
                    if isinstance(arg, immutable_list):
                        logger.debug(
                            f"currently not support fuse node with complex args"
                        )
                        raise RuntimeError("Node with complex args")
            except RuntimeError:
                pass
            fusing_module = GraphModule(
                self.graph_mod, build_sub_graph(self.origin_graph, fusing_nodes)
            )
            try:
                fusing_module.recompile()
                fusing_module.delete_all_unused_submodules()
            except RuntimeError as e:
                logger.debug(f"fusing module recompiling failed")
                logger.debug(f"\\\\\\\\ {e}")
                break
            try:
                JitModuleFactory.produce(fusing_module)
            except ValueError as e:
                logger.debug(f"can't find jit module")
                break
            logger.debug(f"fuse node `%{cur_node.name}`")
            if not is_last_try:
                (cur_node,) = cur_node.users
            else:
                cur_node = None
                fusing_nodes.append(None)
        fusable_nodes = fusing_nodes[:-1]
        if len(fusable_nodes) >= 1:
            # At least one node can be fused
            return fusable_nodes
        else:
            return None

    def fuse_nodes(self, fused_nodes, fuse_parteners_of):
        # Fuse nodes and add them into graph
        origin_graph = self.origin_graph
        logger.debug("start fusing")
        new_graph = Graph()
        node_remap = {}
        inserted_nodes = set()
        for node in origin_graph.nodes:
            if node in inserted_nodes:
                continue
            if node not in fused_nodes:
                node_remap[node] = new_graph.node_copy(node, lambda n: node_remap[n])
                inserted_nodes.add(node)
            else:
                # Generate args for fused node, insert if necessary
                fused_node_args = []
                fused_node_args_sample_inputs = []
                for fpn in fuse_parteners_of[node]:
                    inserted_nodes.add(fpn)
                    for arg in fpn.args:
                        if isinstance(arg, Node):
                            if arg not in fuse_parteners_of[node]:
                                if arg not in node_remap.keys():
                                    node_remap[arg] = new_graph.node_copy(
                                        arg, lambda n: node_remap[n]
                                    )
                                if node_remap[arg] not in fused_node_args:
                                    fused_node_args.append(node_remap[arg])
                                    fused_node_args_sample_inputs.append(
                                        Graph._get_output_from_node_or_list(
                                            node_remap[arg]
                                        )
                                    )
                        else:
                            fused_node_args.append(arg)
                            fused_node_args_sample_inputs.append(arg)
                if len(fused_node_args_sample_inputs) == 1:
                    fused_node_args_sample_inputs = fused_node_args_sample_inputs[0]
                # Build fused module and insert
                fused_module = GraphModule(
                    self.graph_mod,
                    build_sub_graph(origin_graph, fuse_parteners_of[node]),
                )
                fused_module.recompile()
                fused_module.delete_all_unused_submodules()
                # TODO: make module
                try:
                    fused_jit_module = make_jit_module(
                        modules=fused_module,
                        sample_inputs=fused_node_args_sample_inputs,
                    )
                except Exception as e:
                    logger.debug(e)
                    logger.debug(
                        f"Fail to make jit module for nodes {[n.name for n in fuse_parteners_of[node]]}. Is this kernel already tuned?"
                    )
                    for fpn in fuse_parteners_of[node]:
                        if fpn not in node_remap:
                            node_remap[fpn] = new_graph.node_copy(
                                fpn, lambda n: node_remap[n]
                            )
                    continue

                fused_module_target = "BRT_VF__" + "__".join(
                    fpn.name for fpn in fuse_parteners_of[node]
                )
                self.graph_mod.add_submodule(fused_module_target, fused_jit_module)
                fused_node = new_graph.create_node(
                    op="call_module",
                    target=fused_module_target,
                    args=tuple(fused_node_args),
                    name=fused_module_target,
                    is_fixed_inout=True,
                    inshape=Graph._get_shape_from_tensor_or_list(
                        fused_node_args_sample_inputs
                    ),
                    outshape=fuse_parteners_of[node][-1].outshape,
                )
                for fpn in fuse_parteners_of[node]:
                    node_remap[fpn] = fused_node
        self.graph_mod.graph = new_graph
        # Clean
        self.graph_mod.recompile()

    def _generate_fused_inputs(self, fuse_parteners: List[Node]):
        # Generating inputs and outputs
        fused_node_args = []
        fused_node_args_sample_inputs = []
        for fpn in fuse_parteners:
            for arg in fpn.args:
                if isinstance(arg, Node):
                    if arg not in fuse_parteners:
                        # if arg not in node_remap.keys():
                        #     node_remap[arg] = new_graph.node_copy(
                        #         arg, lambda n: node_remap[n]
                        #     )
                        # if node_remap[arg] not in fused_node_args:
                        if arg not in fused_node_args:
                            fused_node_args.append(arg)
                            fused_node_args_sample_inputs.append(
                                Graph._get_output_from_node_or_list(arg)
                            )
                else:
                    fused_node_args.append(arg)
                    fused_node_args_sample_inputs.append(arg)
        if len(fused_node_args_sample_inputs) == 1:
            fused_node_args_sample_inputs = fused_node_args_sample_inputs[0]
        return fused_node_args, fused_node_args_sample_inputs

    def _make_fused_module(self, fuse_parteners: List[Node]):
        args, sample_inputs = self._generate_fused_inputs(fuse_parteners)
        # Build fused module and insert
        vfused_module = GraphModule(
            self.graph_mod,
            build_sub_graph(self.origin_graph, fuse_parteners),
        )
        vfused_module.recompile()
        vfused_module.delete_all_unused_submodules()
        fused_jit_module = make_jit_module(
            modules=vfused_module,
            sample_inputs=sample_inputs,
        )
        return fused_jit_module, args, sample_inputs

    def _replace_with_fused_module(self, fuse_parteners: List[Node]):
        # args, sample_inputs = self._generate_fused_inputs(fuse_parteners)
        # # Build fused module and insert
        # fused_module_graph = build_sub_graph(self.origin_graph, fuse_parteners)
        # self.graph_mod.graph = fused_module_graph
        # self.graph_mod.recompile()
        fused_jit_module, args, sample_inputs = self._make_fused_module(fuse_parteners)
        fused_module_target = "BRT_VF__" + "__".join(fpn.name for fpn in fuse_parteners)
        self.graph_mod.add_submodule(fused_module_target, fused_jit_module)
        self.sub_modules[fused_module_target] = fused_jit_module
        with self.origin_graph.inserting_after(fuse_parteners[-1]):
            fused_node = self.origin_graph.create_node(
                op="call_module",
                target=fused_module_target,
                args=tuple(args),
                name=fused_module_target,
                is_fixed_inout=True,
                inshape=Graph._get_shape_from_tensor_or_list(sample_inputs),
                outshape=fuse_parteners[-1].outshape,
            )
        fuse_parteners[-1].replace_all_uses_with(fused_node)
        return fused_node
        # user_node.args = map_arg(
        #     user_node.args, lambda n: fused_node if n is fuse_parteners[-1] else n
        # )
        # for fp in reversed(fuse_parteners):
        #     self.origin_graph.erase_node(fp)
        # for fpn in fuse_parteners_of[node]:
        #     node_remap[fpn] = fused_node

    def add_annotation_dfs(self, start: Node, source: Node, skiped: Set[Node]):
        # TODO: handle index-changing node not following a scatter, e.g. getitem, index_select
        if start in skiped:
            return
        skiped.add(start)

        getitem_users: Set[Node] = set()
        indexing_users: Set[Node] = set()
        other_users: Set[Node] = set()
        for user in start.users:
            if self.is_function_node(user) and user.target is operator.getitem:
                getitem_users.add(user)
                indexing_users.add(user)
            elif self.is_router_node(user):
                indexing_users.add(user)
            else:
                other_users.add(user)

        if start is source:  # only if is a index-changing node (e.g. getitem, router)
            for indexing_node in indexing_users:
                self.add_annotation_dfs(indexing_node, indexing_node, skiped)
            if other_users:
                with self.origin_graph.inserting_after(start):
                    depacking_node = self.origin_graph.create_node(
                        op="call_function",
                        target=deinit_grid_tensor,
                        args=(start, False),
                    )
                if start.is_fixed_inout:
                    depacking_node.set_inout_shape(start.outshape, start.outshape)
                start.replace_all_uses_with(
                    depacking_node, lambda user: user in other_users
                )
                self.add_annotation_dfs(depacking_node, start, skiped)
        else:
            for other_node in other_users:
                self.add_annotation_dfs(other_node, source, skiped)
            if indexing_users:
                with self.origin_graph.inserting_after(start):
                    packing_node = self.origin_graph.create_node(
                        op="call_function",
                        target=init_grid_tensor_from,
                        args=(start, source),
                    )
                if start.is_fixed_inout:
                    packing_node.set_inout_shape(start.outshape, start.outshape)
                start.replace_all_uses_with(
                    packing_node, lambda user: user in indexing_users
                )
                skiped.add(packing_node)
                for getitem_node in getitem_users:
                    raise NotImplementedError
                    self.add_annotation_dfs(getitem_node, getitem_node, skiped)

    def add_annotation(self):
        skiped: Set[Node] = set()
        for node in self.origin_graph.nodes:
            if node in skiped:
                continue
            if self.is_scatter_node(node):
                self.add_annotation_dfs(node, node, skiped)
            else:
                skiped.add(node)

    def run_on_graph(self):
        visited = set()
        for node in self.origin_graph.nodes:
            fuse_parteners = self.find_fuse_parteners(node, visited)
            if fuse_parteners is None:
                visited.add(node)
                continue
                # node_remap[node] = new_graph.node_copy(node, lambda n: node_remap[n])
            assert len(fuse_parteners) > 0
            try:
                self._replace_with_fused_module(fuse_parteners)
            except Exception as e:
                logger.debug(e)
                logger.debug(
                    f"Fail to make jit module for nodes {[n.name for n in fuse_parteners]}. Is this kernel already tuned?"
                )
                visited.add(fuse_parteners[0])
                # for fpn in fuse_parteners_of[node]:
                #     if fpn not in node_remap:
                #         node_remap[fpn] = new_graph.node_copy(
                #             fpn, lambda n: node_remap[n]
                #         )
                continue
            visited.update(fuse_parteners)

        self.origin_graph._owners = 0
        self.graph_mod.graph = self.origin_graph

        self.finalize()
        self.add_annotation()
