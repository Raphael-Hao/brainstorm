# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union, Dict, Any, Callable

from copy import deepcopy
from hashlib import new
import itertools as itt

import torch
import torch.nn as nn
from torch import fx
from torch.fx.immutable_collections import immutable_list, immutable_dict
from torch.fx.node import map_arg

from brt.runtime import log
from brt.jit import make_jit_module
from brt.jit.modules.factory import JitModuleFactory
from brt.router import ScatterRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.trace.graph import symbolic_trace, GraphModule, Graph, Node

# import brt.passes.fuse_util
from brt.passes.base import PassBase, register_pass
from brt.passes.utils import build_sub_graph

logger = log.get_logger(__file__)


@register_pass("vertical_fuse")
class VerticalFusePass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        sample_inputs: Dict[str, Any],
        **tracer_kwargs,
    ):
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
        logger.info(self.graph_mod.graph)

    def find_fuse_parteners(
        self, start: Node, enable: Callable[[Node], bool] = None
        # , excludes: List[Node] = None
    ) -> Union[List[Node], None]:
        if enable is None:
            enable = lambda n: True

        if not (self.is_module_node(start) and start.is_fixed_inout):
            logger.debug(f"start node `{start.name}` should be a fixed module node")
            return None

        cur_node: Node = start
        fusing_nodes = []
        is_last_try = False
        while not is_last_try:
            fusing_nodes.append(cur_node)
            if not enable(cur_node):
                logger.debug(f"node `%{cur_node.name}` is disabled")
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
            fusing_graph = build_sub_graph(self.origin_graph, fusing_nodes)
            try:
                fusing_graph.lint()
            except RuntimeError as e:
                logger.debug(f"graph lint failed")
                logger.debug(f"\\\\\\\\ {e}")
                break
            self.graph_mod.graph = fusing_graph
            if fusing_graph.eliminate_dead_code():
                logger.debug(f"graph has dead code")
                break
            self.graph_mod.recompile()
            try:
                JitModuleFactory.produce(self.graph_mod)
            except ValueError:
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

    def find_fusable_nodes(self):
        origin_graph = self.origin_graph
        # fusable_nodes_group = []
        # Searching fusable nodes
        visited_nodes = set()
        fused_nodes = set()
        fuse_parteners_of = {}
        for node in origin_graph.nodes:
            fuse_parteners = self.find_fuse_parteners(node, lambda n: n not in visited_nodes)
            if fuse_parteners is None:
                visited_nodes.add(node)
            else:
                assert len(fuse_parteners) > 0
                visited_nodes.update(fuse_parteners)
                fused_nodes.update(fuse_parteners)
                for fp in fuse_parteners:
                    fuse_parteners_of[fp] = fuse_parteners
                # fusable_nodes_group.append(fuse_parteners)
            self.graph_mod.graph = origin_graph
        return fused_nodes, fuse_parteners_of

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
                fused_graph = build_sub_graph(origin_graph, fuse_parteners_of[node])
                self.graph_mod.graph = fused_graph
                self.graph_mod.recompile()
                # TODO: make module
                try:
                    fused_jit_module = make_jit_module(
                        modules=self.graph_mod,
                        sample_inputs=fused_node_args_sample_inputs,
                    )
                except Exception as e:
                    logger.debug(e)
                    logger.info(
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
        # clean
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
        fused_module_graph = build_sub_graph(self.origin_graph, fuse_parteners)
        self.graph_mod.graph = fused_module_graph
        self.graph_mod.recompile()
        fused_jit_module = make_jit_module(
            modules=self.graph_mod,
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
        # user_node.args = map_arg(
        #     user_node.args, lambda n: fused_node if n is fuse_parteners[-1] else n
        # )
        # for fp in reversed(fuse_parteners):
        #     self.origin_graph.erase_node(fp)
        # for fpn in fuse_parteners_of[node]:
        #     node_remap[fpn] = fused_node

    def run_on_graph(self):
        # fused_nodes, fuse_parteners_of = self.find_fusable_nodes()
        # self.fuse_nodes(fused_nodes, fuse_parteners_of)
        visited_nodes = set()
        # fused_nodes = set()
        # fuse_parteners_of = {}
        # node_remap = {}
        # new_graph = Graph()
        for node in self.origin_graph.nodes:
            fuse_parteners = self.find_fuse_parteners(node, lambda n: n not in visited_nodes)
            if fuse_parteners is None:
                visited_nodes.add(node)
                continue
                # node_remap[node] = new_graph.node_copy(node, lambda n: node_remap[n])
            assert len(fuse_parteners) > 0
            try:
                self._replace_with_fused_module(fuse_parteners)
            except Exception as e:
                logger.debug(e)
                logger.info(
                    f"Fail to make jit module for nodes {[n.name for n in fuse_parteners]}. Is this kernel already tuned?"
                )
                visited_nodes.add(fuse_parteners[0])
                # for fpn in fuse_parteners_of[node]:
                #     if fpn not in node_remap:
                #         node_remap[fpn] = new_graph.node_copy(
                #             fpn, lambda n: node_remap[n]
                #         )
                continue
            visited_nodes.update(fuse_parteners)

        self.origin_graph._owners = 0
        self.graph_mod.graph = self.origin_graph
