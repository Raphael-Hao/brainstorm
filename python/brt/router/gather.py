# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import torch
from brt.common import log
from brt.primitive import router

from .base import BaseRouter
from .dispatcher import DefaultDispatcher, FusedDispatcher
from .symbolic import symbolic_gather_route

__all__ = [
    "GatherRouter",
    "SparseGatherRouter",
    "FusedRandomGatherRouter",
]

logger = log.get_logger(__file__)


@router
class GatherRouter(BaseRouter):
    def __init__(self, route_num: int, reduction: str = "add", **kwargs):
        super().__init__(route_num=route_num)
        self.reduction = reduction
        self.dispatcher = DefaultDispatcher(
            route_num=self.route_num, reduction=self.reduction, **kwargs
        )

    def route(
        self,
        inputs: List[torch.Tensor],
        tags: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        route_results, _ = self.dispatcher.combine(inputs, tags, loads)
        return route_results

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        tags: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        route_results, _ = symbolic_gather_route(inputs, tags, loads, self.route_num)
        return route_results


@router
class SparseGatherRouter(GatherRouter):
    def __init__(self, route_num: int, reduction: str = "add", **kwargs):
        super().__init__(route_num=route_num, reduction=reduction, **kwargs)

    def route(
        self,
        inputs: List[torch.Tensor],
        tags: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        route_results, route_tags = self.dispatcher.combine(inputs, tags)
        return route_results, route_tags

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        tags: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return symbolic_gather_route(inputs, tags, 0, self.route_num)


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
