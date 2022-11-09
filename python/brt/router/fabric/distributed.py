# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union

import torch

# pylint: disable=no-name-in-module
from brt._C.router import (
    dispatch_with_dst_indices_1d,
    dispatch_with_dst_indices_2d,
    combine_with_src_indices,
)
import brt.runtime.distributed as brt_dist

# pylint: enable=no-name-in-module
from brt.router.fabric.base import register_fabric
from brt.router.fabric.fused import FusedDispatchFabric, FusedCombineFabric


@register_fabric("distributed_fused_dispatch")
class DistributedFusedDispatchFabric(FusedDispatchFabric):
    def __init__(
        self,
        flow_num: int,
        capacity_padding=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
    ):
        super().__init__(
            flow_num=flow_num,
            capacity_padding=capacity_padding,
            route_logic=route_logic,
            transform=transform,
        )

    def forward(
        self,
        in_flows: List[torch.Tensor],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[torch.Tensor]:
        in_flows = [in_flows] if self.flow_num == 1 else in_flows
        all_out_flows = []
        for flow_idx, in_flow in enumerate(in_flows):
            if self.route_logics[flow_idx] == "1d":
                if self.transforms[flow_idx]:
                    out_flow = dispatch_with_dst_indices_1d(
                        in_flow, route_indices, loads, self.capacity_padding, score
                    )
                else:
                    out_flow = dispatch_with_dst_indices_1d(
                        in_flow, route_indices, loads, self.capacity_padding
                    )
            elif self.route_logics[flow_idx] == "2d":
                in_flow = in_flow.transpose(0, 1).contiguous()
                out_flow = dispatch_with_dst_indices_2d(
                    in_flow, route_indices, loads, self.capacity_padding
                )
            else:
                raise ValueError("route_logic must be 1d or 2d")

            all_out_flows.append(out_flow)
        all_out_flows = all_out_flows[0] if self.flow_num == 1 else all_out_flows
        all_out_flows, out_loads, reorder_indices = brt_dist.group_asymmetry_a2a(
            all_out_flows, loads, locality_aware=True
        )
        return all_out_flows, route_indices, loads, score


@register_fabric("distributed_fused_combine")
class DistributedFusedCombineFabric(FusedCombineFabric):
    def __init__(self, flow_num, sparse, reduction, granularity_padding) -> None:
        assert granularity_padding == False
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=sparse,
            granularity_padding=False,
        )
        self.transform = True

    def forward(
        self,
        in_flows: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        in_flows = [in_flows] if self.flow_num == 1 else in_flows
        route_indices = in_flows[0].route_indices
        loads = in_flows[0].loads
        score = in_flows[0].score
        out_flows = []
        for _flow_idx, in_flow in enumerate(in_flows):

            if self.transform:
                out_flow = combine_with_src_indices(
                    in_flow, route_indices, loads, auto_pad=True, gates=score
                )
            else:
                out_flow = combine_with_src_indices(in_flow, route_indices, loads, None)

            out_flows.append(out_flow)
        out_flows = out_flows[0] if self.flow_num == 1 else out_flows

        return out_flows
