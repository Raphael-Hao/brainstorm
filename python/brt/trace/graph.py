# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, Callable, Any, Tuple, List, Optional, Dict, Type

import operator
import copy

import torch
from torch import nn, fx
from torch import Tensor, Size
from torch.fx.node import Target, Argument, map_arg
from torch.fx.graph import magic_methods

from brt.router import is_router, RouterBase, ScatterRouter
from brt.runtime import log

from brt.trace.node import Node
from brt.trace.leaf_node import is_leaf_node

__all__ = ["symbolic_trace", "GraphTracer", "GraphModule", "Graph"]


logger = log.get_logger(__file__)

class GraphModule(fx.GraphModule):
    pass


class Graph(fx.Graph):
    def __init__(
        self,
        owning_module: Optional[GraphModule] = None,
        tracer_cls: Optional[Type["GraphTracer"]] = None,
    ):
        super().__init__(owning_module, tracer_cls)
        self._root: Node = Node(self, "", "root", "", (), {})
        self._insert = self._root.prepend
        self.is_shape_tracked = False

    def node_copy(
        self, node: fx.Node, arg_transform: Callable[[Node], Argument] = ...
    ) -> Node:
        if isinstance(node, Node):
            args = map_arg(node.args, arg_transform)
            kwargs = map_arg(node.kwargs, arg_transform)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            result_node = self.create_node(
                node.op,
                node.target,
                args,
                kwargs,
                node.name,
                node.type,
                node.is_fixed_inout,
                node.inshape,
                node.outshape,
            )
            result_node.meta = copy.copy(node.meta)
            return result_node
        else:
            assert isinstance(node, fx.Node)
            return super().node_copy(node, arg_transform)

    def create_node(
        self,
        op: str,
        target: Target,
        args: Optional[Tuple[Argument, ...]] = None,
        kwargs: Optional[Dict[str, Argument]] = None,
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        is_fixed_inout: Optional[bool] = False,
        inshape: Union[torch.Size, None, List] = None,
        outshape: Union[torch.Size, None, List] = None,
    ) -> Node:
        assert op in (
            "call_function",
            "call_method",
            "get_attr",
            "call_module",
            "placeholder",
            "output",
        )
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        assert isinstance(args, tuple), "args must be a tuple"
        assert isinstance(kwargs, dict), "kwargs must be a dict"

        candidate = name if name is not None else self._target_to_str(target)
        name = self._graph_namespace.create_name(candidate, None)
        n = Node(
            self,
            name,
            op,
            target,
            args,
            kwargs,
            type_expr,
            is_fixed_inout,
            inshape,
            outshape,
        )

        self._graph_namespace.associate_name_with_obj(name, n)

        self._insert(n)
        self._len += 1

        return n

    @classmethod
    def _get_shape_from_tensor_or_list(
        cls, tensor_or_list: Union[Tensor, Any, List]
    ) -> Union[Size, None, List]:
        if isinstance(tensor_or_list, Tensor):
            return tensor_or_list.shape
        elif isinstance(tensor_or_list, (list, tuple)):
            return [cls._get_shape_from_tensor_or_list(elem) for elem in tensor_or_list]
        else:
            return None

    @classmethod
    def _get_output_from_size_or_list(
        cls,
        size_or_list: Union[Size, None, List],
        tensor_init_func=torch.randn,
        **kwargs,
    ) -> Union[Tensor, None, List]:
        init_kwargs = {"device": "cuda"}
        if kwargs is not None:
            init_kwargs.update(kwargs)
        if size_or_list is None:
            return None
        elif isinstance(size_or_list, Size):
            return tensor_init_func(size_or_list, **init_kwargs)
        else:
            assert isinstance(size_or_list, (list, tuple)), type(size_or_list)
            return [
                cls._get_output_from_size_or_list(elem, tensor_init_func, **init_kwargs)
                for elem in size_or_list
            ]

    @classmethod
    def _get_output_from_node_or_list(
        cls,
        node_or_list: Union[Node, Any, List],
        module: nn.Module = None,
        tensor_init_func=torch.randn,
        **kwargs,
    ) -> Union[Tensor, None, List]:
        init_kwargs = {"device": "cuda"}
        if kwargs is not None:
            init_kwargs.update(kwargs)

        if isinstance(node_or_list, Node):
            node = node_or_list
            if node.op == "get_attr":  # thus fixed
                return getattr(module, node.target, None)
            elif node.is_fixed_inout:  # thus the outshape is not None
                return cls._get_output_from_size_or_list(
                    node.outshape, tensor_init_func, **init_kwargs
                )
            else:
                return None
        elif isinstance(node_or_list, (list, tuple)):
            return [
                cls._get_output_from_node_or_list(elem, tensor_init_func, **init_kwargs)
                for elem in node_or_list
            ]
        else:
            return node_or_list

    @classmethod
    def _get_out_shape_from_node_or_list(
        cls, node_or_list: Union[Node, Any, Tuple[Node]]
    ) -> Union[Size, None, List]:
        if isinstance(node_or_list, Node):
            return node_or_list.outshape
        elif isinstance(node_or_list, (tuple, list)):
            return [cls._get_out_shape_from_node_or_list(elem) for elem in node_or_list]
        else:
            return None


class GraphTracer(fx.Tracer):
    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        tracing_shape: bool = False,
        sample_inputs: Dict[str, Tensor] = None,
        fixed_inputs: bool = False,
    ) -> Graph:
        """Trace the module with shape propagation

        Args:
            sample_inputs (Tensor, List[Tensor], ...): this is required for tracing if
                `tracing_shape` is `True`,. Currently only support positional arguments
                and 1-D array of Tensor.
            fixed_inputs (bool): if `True`, the module inputs are considered as fixed shape,
                otherwise only fixing shape of module attributes and fixable tensors after
                routers.
        """
        fx_graph = super().trace(root, concrete_args)
        graph = Graph(fx_graph.owning_module, fx_graph._tracer_cls)
        output_vals =  graph.graph_copy(fx_graph, {}, return_output_node=True)
        output_val, old_output_val = output_vals
        graph.output(output_val, type_expr=getattr(old_output_val, 'type', None))
        if tracing_shape:
            graph = self.trace_shape(graph, sample_inputs, fixed_inputs)
        return graph

    def trace_shape(
        self, graph: Graph, sample_inputs: Dict[str, Tensor], fixed_inputs: bool
    )-> Graph:
        if sample_inputs is None:
            sample_inputs = {}

        # Fix all fused dispatching routers
        root = self.root
        router_to_node = {}
        fixed_router_info = {}
        all_hooks = []
        for node in graph.nodes:
            assert isinstance(node, Node), type(node)
            if node.op == "call_module" and isinstance(
                root.get_submodule(node.target), ScatterRouter
            ):
                router: ScatterRouter = root.get_submodule(node.target)
                if (
                    router.capture_mode == "max"
                    and "dispatch" in router.fabric_type
                    and router.load_history is not None
                    and all(rl == "1d" for rl in router.fabric.route_logics)
                ):
                    router.capturing = False
                    router_to_node[router] = node
                    fixed_router_info[node] = [None, None]

                    def get_shape_hook(module, input, output):
                        fixed_router_info[router_to_node[module]][
                            0
                        ] = graph._get_shape_from_tensor_or_list(input)
                        fixed_router_info[router_to_node[module]][
                            1
                        ] = graph._get_shape_from_tensor_or_list(output)

                    all_hooks.append(router.register_forward_hook(get_shape_hook))
                continue
        root(**sample_inputs)
        for hook in all_hooks:
            hook.remove()
        for node, (inshape, outshape) in fixed_router_info.items():
            assert isinstance(node, Node), type(node)
            scatter = root.get_submodule(node.target)
            for i, bs in enumerate(scatter.load_history):
                if scatter.fabric.flow_num == 1:
                    outshape[i][0] = bs
                    outshape[i] = torch.Size([int(bs), *outshape[i][1:]])
                else:
                    for flow_outshape in outshape:
                        flow_outshape[i] = torch.Size([int(bs), *flow_outshape[i][1:]])

            node.set_inout_shape(inshape, outshape)

        # Shape propagation
        for node in graph.nodes:
            assert isinstance(node, Node), type(node)
            # TODO:
            if node.kwargs:
                logger.info(f"Currently not support nodes with kwargs (`{node.name}`), the info of kwargs won't be traced")
            if node.op == "placeholder":
                if not fixed_inputs:
                    continue
                if node.target in sample_inputs:
                    shape = getattr(sample_inputs[node.target], "shape", None)
                    if shape is not None:
                        node.set_inout_shape(None, shape)
                elif node.args is not None:
                    shape = getattr(node.args[0], "shape", None)
                    if shape is not None:
                        # This is typically unreachable, unless the module has an argument
                        # with a tensor-type default value
                        node.set_inout_shape(None, shape)
            elif "call_" in node.op:
                if node.op == "call_module" and is_router(
                    root.get_submodule(node.target)
                ):
                    continue
                if all(
                    arg.is_fixed_inout for arg in node.args if isinstance(arg, Node)
                ):
                    # Assert that arg is either a node or a constant
                    node_inputs = graph._get_output_from_node_or_list(node.args)
                    node_kw_inputs = {key: graph._get_output_from_node_or_list(kwarg) for key, kwarg in node.kwargs.items()}
                else:
                    # Can't propagate shape
                    continue
                if node.op == "call_method":
                    func = getattr(torch.Tensor if isinstance(node.args[0], Node) else type(node.args[0]), node.target)
                elif node.op == "call_module":
                    func = root.get_submodule(node.target).forward
                elif node.op == "call_function":
                    func = node.target
                try:
                    node_outputs = func(*node_inputs, **node_kw_inputs)
                except Exception as e:
                    continue
                inshape = graph._get_out_shape_from_node_or_list(node.args)
                outshape = graph._get_shape_from_tensor_or_list(node_outputs)
                node.set_inout_shape(inshape, outshape)
            elif node.op == "get_attr":
                # Assert the attr is a fixed value
                node.set_inout_shape(
                    None,
                    graph._get_shape_from_tensor_or_list(getattr(root, node.target)),
                )
            elif node.op == "output":
                if all(
                    arg.is_fixed_inout for arg in node.args if isinstance(arg, Node)
                ):
                    ioshape = graph._get_out_shape_from_node_or_list(node.args)
                    node.set_inout_shape(ioshape, ioshape)
            elif node.op == "root":
                pass
        graph.is_shape_tracked = True
        return graph

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:

        ##FIXME this is error when we use deepcopy is_router always returns false
        if is_router(m) or is_leaf_node(m):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class GraphModule(fx.GraphModule):
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        class_name: str = "GraphModule",
    ):
        super().__init__(root, graph, class_name)

    @property
    def graph(self) -> Graph:
        return super().graph
    
    @graph.setter
    def graph(self, g : Graph) -> None:
        fx.GraphModule.graph.fset(self, g)



def symbolic_trace(
    m: nn.Module, name=None, tracing_shape: bool = False, **tracer_kwargs
) -> fx.GraphModule:
    """ Trace a module and propagation the node input and output shapes.

    Args:
        m (nn.Module): the module to be traced.
        name (str): the name of the module, default to its classname
        tracing_shape (bool): whether to trace the shape info, default to False
        tracer_kwargs: see `GraphTracer.trace`
    """
    assert isinstance(
        m, nn.Module
    ), "brt provided symbolic_trace only works on nn.Modules"
    tracer = GraphTracer()
    graph = tracer.trace(m, tracing_shape=tracing_shape, **tracer_kwargs)
    name = m.__class__.__name__ if name is None else name
    return GraphModule(tracer.root, graph, name)
