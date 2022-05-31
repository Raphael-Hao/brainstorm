# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import torch
from brt.primitive import router

from .base import BaseRouter
from .dispatcher import DefaultDispatcher, FusedDispatcher
from .symbolic import symbolic_gather_route

__all__ = [
    "GatherRouter",
    "RandomGatherRouter",
    "FusedRandomGatherRouter",
    "TopKGatherRouter",
    "AggregateGatherRouter",
]


def get_gather_router_kind(reduction: str) -> str:
    if reduction == "add":
        return 0
    else:
        return 1


@router
class GatherRouter(BaseRouter):
    def __init__(
        self,
        route_num: int,
        gran_dim: Union[int, List[int]],
        reduction: str = "add",
        dispatcher_cls=DefaultDispatcher,
    ):
        super().__init__(route_num=route_num, gran_dim=gran_dim)
        self.router_kind = get_gather_router_kind(reduction)
        self.reduction = reduction
        self.dispatcher = dispatcher_cls(
            route_num=self.route_num, gran_dim=self.gran_dim, reduction=self.reduction
        )

    def route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        return self.dispatcher.combine(inputs, reverse_indices, loads)

    def symbolic_route(self, inputs, reverse_indices, loads):
        return symbolic_gather_route(
            inputs, reverse_indices, loads, self.router_kind, self.route_num
        )


@router
class RandomGatherRouter(GatherRouter):
    def __init__(
        self, route_num: int, gran_dim: int = None, dispatcher=None, dtype=None
    ):
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
        self.dispatcher = (
            DefaultDispatcher(self.route_num, gran_dim)
            if dispatcher is None
            else dispatcher
        )

    def route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ) -> torch.Tensor:
        route_results = self.dispatcher.combine(
            inputs,
            reverse_indices,
            reverse_shape,
        )
        return route_results

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ):
        return torch.ops.brt.symbolic_gather_route(
            inputs, reverse_indices, reverse_shape, 0, self.route_num
        )


@router
class FusedRandomGatherRouter(GatherRouter):
    def __init__(self, route_num: int, gran_dim: int = None, dtype=None):
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
        self.dispatcher = FusedDispatcher(self.route_num, None)

    def route(
        self,
        inputs: torch.Tensor,
        reverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        route_results = self.dispatcher.combine(inputs, reverse_indices)
        return route_results


@router
class TopKGatherRouter(GatherRouter):
    def __init__(self, route_num: int, gran_dim: int, k=2, dtype=None):
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
        self.k = k

    def route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ):
        if isinstance(reverse_shape, int):
            route_size = reverse_shape
        elif isinstance(reverse_shape, torch.Size):
            route_size = reverse_shape[0]
        else:
            raise ValueError("origin_shape must be a int or torch.Size")

        valid_input = next(_input for _input in inputs if _input is not None)
        assert (
            self.dtype == valid_input.dtype
        ), "Dtype of router and input must be consistent"
        route_results = torch.zeros(
            size=(route_size, self.gran_dim),
            dtype=valid_input.dtype,
            device=valid_input.device,
        )

        if isinstance(reverse_shape, torch.Size):
            return route_results
        else:
            route_results = torch.chunk(route_results, route_size)
            return route_results

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ):
        return torch.ops.brt.symbolic_gather_route(
            inputs, reverse_indices, reverse_shape, self.k, self.route_num
        )


@router
class AggregateGatherRouter(GatherRouter):
    def __init__(self, route_num: int, dtype=None):
        super().__init__(route_num=route_num, dtype=dtype)

    def route(
        self,
        inputs: List[torch.Tensor],
    ):
        inputs = [_input for _input in inputs if _input.numel() > 0]
        result = torch.stack(inputs, dim=0).sum(dim=0)
        return result

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
    ):
        return symbolic_gather_route(inputs, inputs, inputs[0], 3, self.route_num)
