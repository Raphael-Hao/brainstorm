# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union

import numpy as np
import torch
from brt.common import log
from brt.primitive import router

from .base import BaseRouter
from .flow_tensor import FlowTensor, deinit_flow_tensor, init_flow_tensor
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
        in_flows: List[FlowTensor],
    ) -> FlowTensor:
        in_flows = self.verify_in_flow(in_flows)
        out_flow = self.combine(in_flows)
        out_flow = self.verify_out_flow(out_flow)
        return out_flow

    def combine(self, in_flows: List[FlowTensor]) -> Union[FlowTensor, torch.Tensor]:
        """
        Combine the outputs of the routers into the final outputs
        """
        assert len(in_flows) == self.dst_num
        in_flows_data = []
        in_flows_tag = []
        in_flows_load = []
        flow_tags, flow_loads = [], []
        
        for flow in in_flows:
            data, flow_tags, flow_loads, _ = deinit_flow_tensor(flow)
            in_flows_data.append(data)
            in_flows_tag.append(flow_tags.pop())
            in_flows_load.append(flow_loads.pop())
        in_flows_data = torch.cat(in_flows_data, dim=0)
        in_flows_tag = torch.cat(in_flows_tag, dim=0)
        in_flows_load = np.max(in_flows_load)
        in_flows_tags = flow_tags
        in_flows_loads = flow_loads

        route_shape = list(in_flows_data.shape[1:])
        route_size = np.prod(route_shape)

        if in_flows_data.numel() == 0:
            out_flow_data = torch.zeros(
                0, *route_shape, dtype=in_flows_data.dtype, device=in_flows_data.device
            )
            out_flow_tag = in_flows_tag
        else:
            if self.sparse:
                out_flow_tag, inverse = torch.unique(
                    in_flows_tag, sorted=True, return_inverse=True
                )
                route_indices = inverse.repeat(1, route_size).view(-1, *route_shape)
                out_flow_tag = out_flow_tag.view(-1, 1)
                out_flow_load = out_flow_tag.numel()
            else:
                route_indices = in_flows_tag.repeat(1, route_size).view(
                    -1, *route_shape
                )
                out_flow_tag = torch.arange(
                    0, in_flows_load, dtype=torch.int64, device=in_flows_data.device
                ).view(-1, 1)
                out_flow_load = in_flows_load
            out_flow_data = torch.zeros(
                out_flow_load,
                *route_shape,
                dtype=in_flows_data.dtype,
                device=in_flows_data.device
            ).scatter_(0, route_indices, in_flows_data, reduce=self.reduction)
        out_flow_tags = in_flows_tags + [out_flow_tag]
        out_flow_loads = in_flows_loads + [in_flows_load]
        # results_data = torch.scatter_reduce(route_datas, 0, route_indices, self.reduction)
        return init_flow_tensor(out_flow_data, out_flow_tags, out_flow_loads)

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
