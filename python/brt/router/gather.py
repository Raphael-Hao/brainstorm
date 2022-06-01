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
    "FusedRandomGatherRouter",
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

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        return symbolic_gather_route(
            inputs, reverse_indices, loads, self.router_kind, self.route_num
        )


@router
class SparseGatherRouter(BaseRouter):
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

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        return symbolic_gather_route(
            inputs, reverse_indices, loads, self.router_kind, self.route_num
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
