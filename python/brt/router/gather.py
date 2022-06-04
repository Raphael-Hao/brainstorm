# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import numpy as np
import torch
from brt.common import log
from brt.primitive import router

from .base import BaseRouter
from .flow_tensor import FlowTensor
from .symbolic import symbolic_gather_route

__all__ = [
    "GatherRouter",
    "FusedRandomGatherRouter",
]

logger = log.get_logger(__file__)


@router
class GatherRouter(BaseRouter):
    def __init__(self, dst_num: int, reduction: str = "add", sparse=True, **kwargs):
        super().__init__(dst_num=dst_num)
        self.sparse = sparse
        self.reduction = reduction

    def route(
        self,
        inputs: List[FlowTensor],
    ) -> FlowTensor:
        route_results = self.combine(inputs)
        return route_results

    def combine(self, inputs: List[FlowTensor]) -> FlowTensor:
        """
        Combine the outputs of the routers into the final outputs
        """
        assert len(inputs) == self.dst_num
        route_datas = torch.cat(inputs, dim=0)
        route_tags = torch.cat([_input.tag for _input in inputs], dim=0)
        load = np.max([_input.load for _input in inputs])
        route_shape = list(route_datas.shape[1:])
        route_size = np.prod(route_shape)
        if route_datas.numel() == 0:
            result_data = torch.zeros(
                0, *route_shape, dtype=route_datas.dtype, device=route_datas.device
            )
            result_tag = route_tags
        else:
            if self.sparse:
                result_tag, inverse = torch.unique(
                    route_tags, sorted=True, return_inverse=True
                )
                route_indices = inverse.repeat(1, route_size).view(-1, *route_shape)
                load = result_tag.size(0)
            else:
                route_indices = route_tags.repeat(1, route_size).view(-1, *route_shape)
                result_tag = torch.arange(
                    0, load, dtype=torch.int64, device=route_datas.device
                ).view(-1, 1)
            result_data = torch.zeros(
                load, *route_shape, dtype=route_datas.dtype, device=route_datas.device
            ).scatter_(0, route_indices, route_datas, reduce=self.reduction)
        # results_data = torch.scatter_reduce(route_datas, 0, route_indices, self.reduction)
        return FlowTensor(result_data).init_flow(result_tag, load)

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        return symbolic_gather_route(inputs, self.dst_num)


@router
class FusedRandomGatherRouter(GatherRouter):
    def __init__(self, dst_num: int, gran_dim: int = None, dtype=None):
        super().__init__(dst_num=dst_num, gran_dim=gran_dim, dtype=dtype)

    def route(
        self,
        inputs: torch.Tensor,
        reverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        route_results = self.dispatcher.combine(inputs, reverse_indices)
        return route_results
