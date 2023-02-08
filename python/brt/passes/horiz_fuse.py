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
            logger.info(f"Scatter node `{node.name}` founded")
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
                        logger.info(
                            f"At branch {i}, router node `{cur_node.name}` founded"
                        )
                        wavefront.discard(cur_node)
                        continue
                    if not is_at_wavefront(cur_node, visited):
                        logger.info(
                            f"At branch {i}, node `{cur_node.name}` is not at wavefront"
                        )
                        raise RuntimeError()
                        continue
                    fuse_parteners = self.find_fuse_parteners(cur_node, visited)
                    # The `cur_node` is unfusable, go ahead
                    if fuse_parteners is None:
                        # TODO: assume there is no branch cross other than the gather router
                        logger.info(
                            f"At branch {i}, node `{cur_node.name}` is not vfusable"
                        )
                        visited.add(cur_node)
                        update_wavefront(cur_node, visited, wavefront)
                        continue
                    # Try to make fused module
                    try:
                        VerticalFusePass._make_fused_module(self, fuse_parteners)
                    except Exception as e:
                        logger.info(
                            f"At branch {i}, fail to make vfused jit module for nodes {[n.name for n in fuse_parteners]}. Is this kernel already tuned?",
                        )
                        visited.add(cur_node)
                        update_wavefront(cur_node, visited, wavefront)
                        continue
                    logger.info(
                        f"At branch {i}, vfusable nodes founded, {[fp.name for fp in fuse_parteners]}"
                    )
                    visited.update(fuse_parteners)
                    update_wavefront(fuse_parteners[-1], visited, wavefront)
                    hfusable_nodes.append(fuse_parteners)
                if len(hfusable_nodes) == 0:
                    logger.info(f"No nodes are h-fusable, continue")
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
                logger.info(f"add h-fused module `{hfused_module_target}`")
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
                    logger.info(f"create node `{get_item_node.name}`")
                    fuse_parteners[-1].replace_all_uses_with(get_item_node)
                # for fuse_parteners in hfusable_nodes:
                #     for fp in reversed(fuse_parteners):
                #         self.origin_graph.erase_node(fp)

        self.origin_graph._owners = 0
        self.graph_mod.graph = self.origin_graph

    def _old_run_on_graph(self):
        ## TODO
        self.graph_mod.graph.eliminate_dead_code()

        def BFS(s):
            queue = [s]
            outList = []
            seen = set()
            node_visited = {}
            for node_1 in self.graph_mod.graph.nodes:
                node_visited.update({node_1: 0})
            while queue:
                ## begin a new level
                res = []
                nextQueue = []
                fuse_sequece = []
                fuse_inputs_shape = []
                fuse_outputs_shape = []
                fuse_target = ""
                fuse_graphnode = []
                for i in range(len(queue)):
                    point = queue[i]
                    res.append(point)
                    node_visited.update({point: 1})
                    if point.op == "call_module":
                        node_m = self.sub_modules[point.target]
                        if isinstance(node_m, nn.Conv2d) and node_m.bias is None:
                            if len(point.users) > 1:
                                print("conv2d has users")
                                continue
                            (bn,) = point.users
                            if len(bn.users) > 1:
                                print("bn has users")
                                continue
                            (activate,) = bn.users
                            node_visited.update({activate: 1})
                            node_visited.update({bn: 1})
                            bn_module = self.sub_modules[bn.target]
                            activate_module = self.sub_modules[activate.target]
                            sequence = nn.Sequential(node_m, bn_module, activate_module)
                            fuse_inputs_shape.append(list(node_m.input_shape))
                            fuse_outputs_shape.append(list(node_m.output_shape))
                            fuse_sequece.append(sequence)
                            fuse_graphnode.append([point, bn, activate])
                            fuse_target += point.target
                            queue[i] = activate
                if len(fuse_sequece) > 0:
                    fuselayer = brt.passes.fuse_util.FusedLayer(
                        fuse_sequece, fuse_inputs_shape, fuse_outputs_shape
                    )
                    construct_new_args = []
                    for i, seq in enumerate(fuse_sequece):
                        construct_new_args.append(fuse_graphnode[i][0].args[0])
                    pre_node = outList[-1][-1]
                    with self.graph_mod.graph.inserting_after(pre_node):
                        new_args = (construct_new_args,)
                        self.graph_mod.add_module(
                            fuse_target.replace(".", "_"), fuselayer
                        )
                        self.sub_modules[fuse_target.replace(".", "_")] = fuselayer
                        new_node_insert = self.graph_mod.graph.create_node(
                            "call_module",
                            fuse_target.replace(".", "_"),
                            args=new_args,
                            kwargs={},
                        )
                    res.append(new_node_insert)

                for i in range(len(queue)):
                    point = queue[i]
                    for son in point.users:
                        if (
                            son.target == torch.cat
                            or self.is_scatter_node(son)
                            or self.is_gather_node(son)
                            or son.target == "view"
                        ):
                            visitall_cat_arg = True
                            if son.target == "view":
                                iterargs = son.args[:2]
                            else:
                                iterargs = son.args[0]
                            for cat_fa in iterargs:
                                if node_visited[cat_fa] == 0:
                                    visitall_cat_arg = False

                            if visitall_cat_arg and node_visited[son] == 0:
                                node_visited.update({son: 1})
                                if self.is_scatter_node(son):
                                    assert len(queue) == 1
                                    for git in son.users:
                                        for git2 in git.users:
                                            nextQueue.append(git2)
                                    break
                                for son_son in son.users:
                                    if (
                                        son_son.target == torch.cat
                                        or self.is_scatter_node(son_son)
                                    ):
                                        continue
                                    nextQueue.append(son_son)
                            continue
                        nextQueue.append(son)
                outList.append(res)
                queue = nextQueue
            return outList

        for node in self.graph_mod.graph.nodes:
            out_list = BFS(node)
            break
        for i in range(len(out_list)):
            fuse_sequece = []
            fuse_inputs_shape = []
            fuse_outputs_shape = []
            fuse_target = ""
            fuse_graphnode = []
            queue = out_list[i]
            for j in range(len(queue)):
                point = queue[j]
                if point.op == "call_module":
                    node_m = self.sub_modules[point.target]
                    if isinstance(node_m, nn.Conv2d) and node_m.bias is None:
                        if len(point.users) > 1:
                            print("conv2d has users")
                            continue
                        (bn,) = point.users
                        if len(bn.users) > 1:
                            print("bn has users")
                            continue
                        (activate,) = bn.users
                        bn_module = self.sub_modules[bn.target]
                        activate_module = self.sub_modules[activate.target]
                        sequence = nn.Sequential(node_m, bn_module, activate_module)
                        fuse_inputs_shape.append(list(node_m.input_shape))
                        fuse_outputs_shape.append(list(node_m.output_shape))
                        fuse_sequece.append(sequence)
                        fuse_graphnode.append([point, bn, activate])
                        fuse_target += point.target
            if len(fuse_sequece) > 0:
                for j, seq in enumerate(fuse_sequece):
                    conv = fuse_graphnode[j][0]
                    bn = fuse_graphnode[j][1]
                    activate = fuse_graphnode[j][2]
                    node_insert = out_list[i][-1]
                    with self.graph_mod.graph.inserting_after(node_insert):
                        new_gititem = self.graph_mod.graph.call_function(
                            operator.getitem, args=(node_insert, j)
                        )
                    activate.replace_all_uses_with(new_gititem)
                    self.graph_mod.graph.erase_node(activate)
                    self.graph_mod.graph.erase_node(bn)
                    self.graph_mod.graph.erase_node(conv)

        self.graph_mod.recompile()

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
