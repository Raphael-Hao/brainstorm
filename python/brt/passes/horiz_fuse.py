# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union, Dict, Any, Set, Callable, Tuple

import operator
import itertools as itt

import numpy as np

import torch
from torch import fx
from torch import nn

import brt
from brt.runtime import log
from brt.router import ScatterRouter, GatherRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.trace.graph import symbolic_trace, GraphModule, Graph, Node
from brt.jit import make_jit_module

from brt.passes.base import PassBase, register_pass
from brt.passes.vertical_fuse import VerticalFusePass
from brt.passes.utils import *

logger = log.get_logger(__file__)


def bfs_succ_list(node: Node):
    succ_list = []
    node_queue = list(node.users)
    while len(node_queue) > 0:
        cur_node = node_queue.pop(0)
        if not all(
            arg in succ_list or not isinstance(arg, Node) for arg in cur_node.args
        ):
            logger.debug(f"Node {cur_node} is supposed to be a gather router.")
            continue
        node_queue.extend(cur_node.users)
        succ_list.append(cur_node)
        yield cur_node


def bfs_succ(node: Node, excludes: Set[Node] = None):
    if excludes is None:
        excludes = set()
    node_queue = list(node.users)
    while len(node_queue) > 0:
        cur_node = node_queue.pop(0)
        if cur_node not in excludes and is_at_wavefront(cur_node, excludes):
            return cur_node
        node_queue.extend(cur_node.users)
    return None


def topo_succ(node: Node, visited: Set[Node] = None):
    if visited is None:
        visited = set()
    wavefront = set()
    wavefront.update(node.users.keys())
    root = node.graph._root
    cur = node._next
    while True:
        if cur is root:
            return None
        if cur not in visited and cur in wavefront:
            return cur
        else:
            if cur in visited:
                wavefront.update(cur.users.keys())
            cur = cur._next


@register_pass("horiz_fuse")
class HorizFusePass(VerticalFusePass):
    def run_on_graph(self):
        visited = set()  # visited nodes in the origin graph

        for node in self.origin_graph.nodes:
            visited.add(node)
            if not (self.is_scatter_node(node) and node.is_fixed_inout):
                continue
            logger.debug(f"Scatter node `{node.name}` founded")
            wavefront = set()
            update_wavefront(node, visited, wavefront)
            while wavefront:
                hfusable_nodes = []
                # Try to fuse node horizontally layer by layer
                # for i, cur_node in enumerate(wavefront):
                cur_nodes = list(wavefront)
                for i, cur_node in enumerate(cur_nodes):
                    if cur_node is None:
                        raise RuntimeError()
                        continue
                    if self.is_router_node(cur_node):
                        logger.debug(
                            f"At branch {i}, router node `{cur_node.name}` founded"
                        )
                        wavefront.discard(cur_node)
                        continue
                    if not is_at_wavefront(cur_node, visited):
                        logger.debug(
                            f"At branch {i}, node `{cur_node.name}` is not at wavefront"
                        )
                        raise RuntimeError()
                        continue
                    fuse_parteners = self.find_fuse_parteners(cur_node, visited)
                    # The `cur_node` is unfusable, go ahead
                    if fuse_parteners is None:
                        # TODO: assume there is no branch cross other than the gather router
                        logger.debug(
                            f"At branch {i}, node `{cur_node.name}` is not vfusable"
                        )
                        visited.add(cur_node)
                        update_wavefront(cur_node, visited, wavefront)
                        continue
                    # Try to make fused module
                    try:
                        VerticalFusePass._make_fused_module(self, fuse_parteners)
                    except Exception as e:
                        logger.debug(
                            f"At branch {i}, fail to make vfused jit module for nodes {[n.name for n in fuse_parteners]}. Is this kernel already tuned?",
                        )
                        visited.add(cur_node)
                        update_wavefront(cur_node, visited, wavefront)
                        continue
                    logger.debug(
                        f"At branch {i}, vfusable nodes founded, {[fp.name for fp in fuse_parteners]}"
                    )
                    visited.update(fuse_parteners)
                    update_wavefront(fuse_parteners[-1], visited, wavefront)
                    hfusable_nodes.append(fuse_parteners)
                if len(hfusable_nodes) == 0:
                    logger.debug(f"No nodes are h-fusable, continue")
                    continue
                # Build h-fuse candidates
                hfusable_modules = nn.ModuleList()
                for fuse_parteners in hfusable_nodes:
                    hfusable_candidate = GraphModule(
                        self.graph_mod,
                        build_sub_graph(self.origin_graph, fuse_parteners),
                    )
                    hfusable_candidate.recompile()
                    hfusable_modules.append(hfusable_candidate)
                # Generate h-fused inputs
                hfused_sample_inputs = []
                hfused_args = []
                for fuse_parteners in hfusable_nodes:
                    (
                        vfused_node_args,
                        vfused_node_args_sample_inputs,
                    ) = VerticalFusePass._generate_fused_inputs(self, fuse_parteners)
                    hfused_sample_inputs.append(vfused_node_args_sample_inputs)
                    hfused_args.extend(vfused_node_args)
                # Build h-fused module and node
                try:
                    hfused_jit_module = make_jit_module(
                        hfusable_modules, hfused_sample_inputs, opt_level="horiz_fuse"
                    )
                except:
                    # NOTE: Remember to handle the visited_nodes
                    logger.warning(
                        f"Horizontal fusion failed. There might be something wrong!",
                        exc_info=True,
                    )
                hfused_module_target = "BRT_HF__V_" + "__V_".join(
                    "__".join(fpn.name for fpn in fuse_parteners)
                    for fuse_parteners in hfusable_nodes
                )
                logger.debug(f"add h-fused module `{hfused_module_target}`")
                self.graph_mod.add_submodule(hfused_module_target, hfused_jit_module)
                self.sub_modules[hfused_module_target] = hfused_jit_module
                with self.origin_graph.inserting_before():
                    hfused_node = self.origin_graph.create_node(
                        op="call_module",
                        target=hfused_module_target,
                        args=tuple(hfused_args),
                        name=hfused_module_target,
                        is_fixed_inout=True,
                        inshape=Graph._get_shape_from_tensor_or_list(
                            itt.chain.from_iterable(hfused_sample_inputs)
                        ),
                        outshape=[
                            fuse_parteners[-1].outshape
                            for fuse_parteners in hfusable_nodes
                        ],
                    )
                visited.add(hfused_node)

                for i, fuse_parteners in enumerate(hfusable_nodes):
                    with self.origin_graph.inserting_after(hfused_node):
                        get_item_node = self.origin_graph.create_node(
                            op="call_function",
                            target=operator.getitem,
                            args=(hfused_node, i),
                            # name="BRT_getitem",
                            is_fixed_inout=True,
                            inshape=hfused_node.outshape,
                            outshape=hfused_node.outshape[i],
                        )
                    visited.add(get_item_node)
                    logger.debug(f"create node `{get_item_node.name}`")
                    fuse_parteners[-1].replace_all_uses_with(get_item_node)
                # for fuse_parteners in hfusable_nodes:
                #     for fp in reversed(fuse_parteners):
                #         self.origin_graph.erase_node(fp)

        self.origin_graph._owners = 0
        self.graph_mod.graph = self.origin_graph

    def finalize(self) -> GraphModule:
        def topological_inner(cur: Node, visited: Set[Node], sequence: List[Node]):
            assert cur not in visited
            visited.add(cur)
            for user in cur.users:
                if user not in visited:
                    topological_inner(user, visited, sequence)
            sequence.append(cur)

        def topoligical_sort(graph: Graph):
            # Get topoligically ordered sequence
            visited = set()
            sequence = []
            placeholders = []
            for node in graph.nodes:
                if self.is_placeholder_node(node):
                    visited.add(node)
                    placeholders.append(node)
                    continue
                if node not in visited:
                    topological_inner(node, visited, sequence)
            # Make topoligically link list
            pre_node = graph._root
            for node in itt.chain(placeholders, reversed(sequence)):
                pre_node._next = node
                node._prev = pre_node
                pre_node = node
            graph._root._prev = pre_node
            pre_node._next = graph._root

        topoligical_sort(self.graph_mod.graph)
        # self.graph_mod.graph.eliminate_dead_code()
        return super().finalize()
