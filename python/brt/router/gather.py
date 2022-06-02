# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import torch
from brt.common import log
from brt.primitive import router

from .base import BaseRouter
from .dispatcher import DefaultDispatcher, FusedDispatcher
from .symbolic import symbolic_gather_route

__all__ = [
    "GatherRouter",
    "FusedRandomGatherRouter",
]

logger = log.get_logger(__file__)

@router
class GatherRouter(BaseRouter):
    def __init__(
        self,
        route_num: int,
        route_method:str="truth",
        reduction: str = "add",
        dispatcher_cls=DefaultDispatcher,
        **kwargs
    ):
        super().__init__(route_num=route_num)
        self.route_method = route_method
        self.reduction = reduction
        self.dispatcher = dispatcher_cls(
            route_num=self.route_num, reduction=self.reduction, **kwargs
        )

    def route(
        self,
        inputs: List[torch.Tensor],
        tags: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        route_results = self.dispatcher.combine(inputs, tags, loads)
        if self.route_method == "restore" and route_results.numel() == 0:
            route_results = torch.zeros(
                loads,
                *list(inputs[0].shape)[1:],
                dtype=inputs[0].dtype,
                device=inputs[0].device,
            )
        return route_results

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        tags: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        return symbolic_gather_route(inputs, tags, loads, self.route_num)


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
        return symbolic_gather_route(inputs, reverse_indices, loads, self.route_num)


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
