# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Union,
    Callable,
    Any,
    Tuple,
    List,
    Optional,
    Dict,
    Set,
    Type,
)

import torch
import torch.fx as fx
from torch.fx.node import Target, Argument

if TYPE_CHECKING:
    from brt.trace.graph import Graph


class Node(fx.Node):
    def __init__(
        self,
        graph: Graph,
        name: str,
        op: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        return_type: Optional[Any] = None,
        is_fixed_inout: Optional[bool] = False,
        inshape: Union[torch.Size, None, List] = None,
        outshape: Union[torch.Size, None, List] = None,
    ) -> None:
        """Appending torch.fx.Node to enable shape propagation.

        Members:
            is_fixed_inout (bool): True if the node has fixed inputs and outputs. 'Fixed' means
                that the value is (a list of) either constant (e.g. a attribute) or a tensor
                with a fixed shape.
            inshape (torch.Size, None, List): (for each member if a list) a `torch.Size` instance
                only if the dependent node is fixed and without a None `outshape`.
            outshape (torch.Size, None, List): (for each member if a list) a `torch.Size` instance
                only if the return value is a tensor with fixed shape, unless it's constant.
        """
        super().__init__(graph, name, op, target, args, kwargs, return_type)
        # True if the node is after a router with fixable cell distrubution
        self.is_fixed_inout = is_fixed_inout
        self.inshape: Union[torch.Size, None, List] = inshape
        self.outshape: Union[torch.Size, None, List] = outshape

    def set_inout_shape(
        self,
        inshape: Union[torch.Size, None, List],
        outshape: Union[torch.Size, None, List],
    ):
        self.is_fixed_inout = True
        self.inshape = inshape
        self.outshape = outshape
        # print(f"[DEBUG] {self.name} | {self.inshape} | {self.outshape}")

    def unset_inout_shape(self):
        self.is_fixed_inout = False
        self.inshape = None
        self.outshape = None

    def format_node(
        self,
        placeholder_names: Optional[List[str]] = None,
        maybe_return_typename: Optional[List[str]] = None,
    ) -> Optional[str]:
        name = super().format_node(placeholder_names, maybe_return_typename)
        if self.graph is not None and self.graph.is_shape_tracked:
            name += f" | {'fixed' if self.is_fixed_inout else 'unfixed'}"
        return name
