# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union, Dict, Any

from copy import deepcopy
from hashlib import new
import itertools as itt

import torch
import torch.nn as nn
from torch import fx
from torch.fx.immutable_collections import immutable_list, immutable_dict

from brt.runtime import log
from brt.jit import make_jit_module
from brt.jit.modules.factory import JitModuleFactory
from brt.router import ScatterRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.trace.graph import symbolic_trace, GraphModule, Graph, Node

# import brt.passes.fuse_util
from brt.passes.base import PassBase, register_pass

logger = log.get_logger(__file__)

def _build_sub_graph(
    graph: Graph, nodes: List[Node], outputs: Union[List[Node], Node, int, None] = -1
) -> Graph:
    """Build a subgraph with `nodes` from `graph`, also generate inputs and outputs.

    Args:
        graph (Graph): The graph building from.
        nodes (List[Node]): The node of new subgraph.
        outputs (Union[List[Node], Node, int, None], optional): The return value node. Defaults to -1.
            None: Return None.
            int: Return `nodes[outputs]`.
            Node | List[Node]: Return `outputs`.

    Returns:
        Graph: the subgraph
    """
    new_graph = Graph()
    node_remap = {}
    # insert nodes
    for node in nodes:
        assert len(node.kwargs) == 0, node.kwargs
        for arg in node.args:
            if isinstance(arg, Node) and arg not in node_remap:
                # node_remap[arg] = new_graph.placeholder(arg.name, arg.type)
                node_remap[arg] = new_graph.create_node(
                    "placeholder",
                    arg.name,
                    name=arg.name,
                    type_expr=arg.type,
                    is_fixed_inout=arg.is_fixed_inout,
                    inshape=arg.inshape,
                    outshape=arg.outshape,
                )
        node_remap[node] = new_graph.node_copy(node, lambda n: node_remap[n])
    # add output
    if outputs is not None:
        if isinstance(outputs, int):
            outputs = nodes[outputs]
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
        outputs = tuple(node_remap[o] for o in outputs)
        new_graph.create_node(
            op="output",
            target="output",
            args=outputs,
            name="output",
        )
    return new_graph


# def _fuse_nodes_into(nodes: List[Node], module_name: str, to: Graph):
#     # fused_node = Node()
#     fused_node_args = []
#     fused_node_kwargs = {}
#     for node in nodes:
#         if node.op in "placeholder":
#             # node_remap[node] = fused_graph.node_copy(node)
#             continue
#         elif node.op == "output":
#             continue
#         else:
#             for arg in node.args:
#                 if arg.op == "placeholder" and arg not in fused_node_args:
#                     fused_node_args.append(arg)
#             for k, n in fused_node_kwargs.items():
#                 if k not in fused_node_kwargs:
#                     fused_node_kwargs[k] = n
#                 else:
#                     raise RuntimeError(f"same keyword `{k}` args in nodes to be fused")
#     fused_node = to.call_module(
#         module_name=module_name, args=tuple(fused_node_args), kwargs=fused_node_kwargs,
#     )
#     return fused_node


# def _combine_graph(into: Graph, another: Graph):
#     # input_node_names = []
#     name_to_node = {}
#     for node in into.nodes:
#         # if node.op == "placeholder":
#         #     input_node_names.append(node.name)
#         if node.op == "output":
#             raise
#         name_to_node[node.name] = node
#     node_remap = {}
#     for node in another.nodes:
#         if node.op == "placeholder":
#             if node.name in name_to_node:
#                 node_remap[node] = name_to_node[node.name]
#             else:
#                 node_remap[node] = into.node_copy(node, lambda n: node_remap[n])
#         elif node.op == "output":
#             continue
#         else:
#             if node.name in name_to_node:
#                 same_node = name_to_node[node.name]
#                 if same_node.op != "placeholder":
#                     raise ValueError(
#                         f"a non-placeholder node is both in `another` `into`"
#                     )
#                 with into.inserting_after(same_node):
#                     node_remap[node] = into.node_copy(node, lambda n: node_remap[n])
#                 same_node.replace_all_uses_with(node_remap[node])
#                 into.erase_node(same_node)
#                 node_remap[node].name = same_node.name
#             else:
#                 node_remap[node] = into.node_copy(node, lambda n: node_remap[n])


@register_pass("vertical_fuse")
class VerticalFusePass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        sample_inputs: Dict[str, Any],
        fixing_scatters: bool = False,
        **tracer_kwargs,
    ):
        self.sub_modules = dict(m.named_modules())
        if fixing_scatters:
            for subm in self.sub_modules.values():
                if isinstance(subm, ScatterRouter):
                    router: ScatterRouter = subm
                    if (
                        router.capturing is True
                        and "dispatch" in router.captured_fabric_type
                        and router.capture_mode == "max"
                        and all(rl == "1d" for rl in router.fabric.route_logics)
                    ):
                        if (
                            router.fabric_type == "dispatch"
                            and router.load_history is not None
                        ):
                            router.fabric_type = "_fused_dispatch"
                            router.fabric_kwargs.update(
                                {
                                    "fixed_capacity": torch.from_numpy(
                                        router.load_history
                                    )
                                    .to(torch.int32)
                                    .cuda()
                                }
                            )
                            router.fabric = make_fabric(
                                "_fused_dispatch", router.fabric_kwargs
                            )
        if isinstance(m, GraphModule) and m.graph.is_shape_tracked:
            m.graph.lint()
            m.recompile()
            self.graph_mod = m
        else:
            self.graph_mod = symbolic_trace(
                m, tracing_shape=True, sample_inputs=sample_inputs, **tracer_kwargs
            )
        logger.info(self.graph_mod.graph)

    def run_on_graph(self):
        origin_graph = self.graph_mod.graph
        # working_graphmod = deepcopy(self.graph_mod)
        # working_graph = working_graphmod.graph
        # working_graph: Graph = Graph().graph_copy(self.graph_mod.graph)
        fusable_nodes_group = []
        # Search fusable nodes
        for node in origin_graph.nodes:
            # May add `visited` field to node
            if getattr(node, "fused", False):
                logger.debug(f"node `{node.name}` already fused")
                continue
            fusing_nodes = []
            if (
                self.is_module_node(node) and node.is_fixed_inout
            ):  # start with module node
                cur_node: Node = node
                while True:
                    fusing_nodes.append(cur_node)
                    if getattr(cur_node, "visited", False):
                        logger.debug(f"node `{cur_node.name}` already visit"
                        )
                        break
                    if self.is_router_node(cur_node):
                        logger.debug(f"router node `{cur_node.name}` found")
                        break
                    if not cur_node.is_fixed_inout:
                        logger.debug(f"node `{cur_node.name}` is not fixed")
                        break
                    if not (
                        self.is_get_attr_node(cur_node)
                        or isinstance(cur_node.outshape, torch.Size)
                    ):
                        logger.debug(f"node `{cur_node.name}` has >1 outputs")
                        break
                    if len(cur_node.users) > 1:
                        logger.debug(
                            f"node `{cur_node.name}` has {len(cur_node.users)} users"
                        )
                        break
                    if len(cur_node.kwargs) != 0:
                        logger.debug(
                            f"node `{cur_node.name}` has kwargs {cur_node.kwargs}"
                        )
                        break
                    try:
                        for arg in cur_node.args:
                            if isinstance(arg, (immutable_list, immutable_dict)):
                                logger.debug(
                                    f"currently not support fuse node with complex args"
                                )
                                raise RuntimeError("Node with complex args")
                    except RuntimeError:
                        pass
                    fusing_graph = _build_sub_graph(origin_graph, fusing_nodes)
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
                        logger.debug(f"finding jit module failed")
                        break
                    logger.debug(f"fuse node `{cur_node.name}`")
                    (cur_node,) = cur_node.users
            fusable_nodes = fusing_nodes[:-1]
            if len(fusable_nodes) >= 1:
                # At least one node can be fused
                for n in fusable_nodes:
                    n.visited = True
                    n.fused = True
                    n.fuse_parteners = fusable_nodes
                fusable_nodes_group.append(fusable_nodes)
            else:
                node.visited = True
                node.fused = False

        # Fuse nodes and add them into graph
        logger.debug("start fusing")
        new_graph = Graph()
        node_remap = {}
        for node in origin_graph.nodes:
            if getattr(node, "inserted", False):
                continue
            if not node.fused:
                node_remap[node] = new_graph.node_copy(node, lambda n: node_remap[n])
                node.inserted = True
            else:
                # Generate args for fused node, insert if necessary
                fused_node_args = []
                fused_node_args_sample_inputs = []
                for fpn in node.fuse_parteners:
                    fpn.inserted = True
                    for arg in fpn.args:
                        if isinstance(arg, Node):
                            if arg not in node.fuse_parteners:
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
                if len(fused_node_args_sample_inputs) == 1:
                    fused_node_args_sample_inputs = fused_node_args_sample_inputs[0]
                    # for key, arg in fpn.kwargs.items():
                    #     if key in fused_node_kwargs:
                    #         raise RuntimeError(
                    #             f"same keyword `{key}` args in nodes to be fused"
                    #         )
                    #     if isinstance(arg, Node):
                    #         if arg not in node.fuse_parteners:
                    #             if arg not in node_remap.keys():
                    #                 node_remap[arg] = new_graph.node_copy(
                    #                     arg, lambda n: node_remap[n]
                    #                 )
                    #             fused_node_kwargs[key] = node_remap[arg]
                    #     else:
                    #         fused_node_kwargs[key] = arg
                # Build fused module and insert
                fused_graph = _build_sub_graph(origin_graph, node.fuse_parteners)
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
                        f"Fail to make jit module for nodes {[n.name for n in node.fuse_parteners]}. Is this kernel already tuned?"
                    )
                    for n in node.fuse_parteners:
                        if n not in node_remap:
                            node_remap[n] = new_graph.node_copy(
                                n, lambda n: node_remap[n]
                            )
                    continue

                fused_module_target = "fused:" + "&".join(
                    fpn.name for fpn in node.fuse_parteners
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
                    outshape=node.fuse_parteners[-1].outshape,
                )
                for fpn in node.fuse_parteners:
                    node_remap[fpn] = fused_node
        self.graph_mod.graph = new_graph
        # clean
        # self.graph_mod.graph.lint()
        self.graph_mod.recompile()
        # node_m = self.sub_modules[node.target]
        # if isinstance(node_m, nn.Conv2d):

        #     if len(node.users) > 1:
        #         print("conv2d has users")
        #         continue
        #     (bn,) = node.users
        #     if len(bn.users) > 1:
        #         print("bn has users")
        #         continue
        #     if node_m.bias is not None:
        #         continue
        #     (activate,) = bn.users
        #     bn_module = self.sub_modules[bn.target]
        #     activate_module = self.sub_modules[activate.target]

        #     sequence = nn.Sequential(node_m, bn_module, activate_module)
        #     new_module = brt.passes.fuse_util.TunedKernel(
        #         sequence, list(node_m.input_shape), list(node_m.output_shape)
        #     )
        #     self.sub_modules[node.target] = new_module
        #     activate.replace_all_uses_with(node)
        #     self.graph_mod.graph.erase_node(activate)
        #     self.graph_mod.graph.erase_node(bn)
        #     with self.graph_mod.graph.inserting_before(node):
        #         self.graph_mod.add_module(
        #             node.target.replace(".", "_"), new_module
        #         )
        #         new_node_insert = self.graph_mod.graph.create_node(
        #             "call_module",
        #             node.target.replace(".", "_"),
        #             args=node.args,
        #             kwargs=node.kwargs,
        #         )
        #         node.replace_all_uses_with(new_node_insert)

        #     self.graph_mod.graph.erase_node(node)

    def finalize(self) -> GraphModule:
        # self.graph_mod.graph.lint()
        return super().finalize()
