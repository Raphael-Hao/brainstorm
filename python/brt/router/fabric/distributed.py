# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union

import brt.runtime.distributed as brt_dist
import torch

# pylint: disable=no-name-in-module
from brt._C.router import (
    combine_with_src_indices,
    dispatch_with_dst_indices_1d,
    dispatch_with_dst_indices_2d,
)

# pylint: enable=no-name-in-module
from brt.router.fabric.base import register_fabric
from brt.router.fabric.fused import FusedCombineFabric, FusedDispatchFabric


@register_fabric("distributed_fused_dispatch")
class DistributedFusedDispatchFabric(FusedDispatchFabric):
    def __init__(
        self,
        flow_num: int,
        capacity_padding=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        locality_aware: bool = False,
    ):
        self.locality_aware = locality_aware
        super().__init__(
            flow_num=flow_num,
            capacity_padding=capacity_padding,
            route_logic=route_logic,
            transform=transform,
        )
        self.register_buffer("placement_indices", None)

    def forward(
        self,
        in_flow: torch.Tensor,
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[torch.Tensor]:
        if self.route_logics[0] == "1d":
            if self.transforms[0]:
                out_flow = dispatch_with_dst_indices_1d(
                    in_flow, route_indices, loads, self.capacity_padding, score
                )
            else:
                out_flow = dispatch_with_dst_indices_1d(
                    in_flow, route_indices, loads, self.capacity_padding
                )
        elif self.route_logics[0] == "2d":
            in_flow = in_flow.transpose(0, 1).contiguous()
            out_flow = dispatch_with_dst_indices_2d(
                in_flow, route_indices, loads, self.capacity_padding
            )
        else:
            raise ValueError("route_logic must be 1d or 2d")

        if self.placement_indices is not None:  # pylint: disable=E0203
            self.placement_indices = self.placement_indices.to(out_flow.device)
            path_num = self.placement_indices.shape[0]
            print("path_num", path_num)
            origin_shape = out_flow.shape
            print("origin_shape", origin_shape)
            out_flow = out_flow.view(path_num, -1)
            print("out_flow.shape", out_flow.shape)
            out_flow = out_flow.index_select(0, self.placement_indices)
            print("out_flow.shape", out_flow.shape)
            out_flow = out_flow.view(origin_shape)
            loads = loads.index_select(0, self.placement_indices)
            route_indices = route_indices.index_select(0, self.placement_indices)
            score = score.index_select(0, self.placement_indices)
            input()

        print("out_flow.shape", out_flow.shape)
        print("loads", loads)
        nodes_results = out_flow.chunk(2)
        print(f"layer id: {self.expert_key} before a2a outflow: {out_flow}")

        a2a_resuslts = brt_dist.group_asymmetry_a2a(
            out_flow, loads, self.locality_aware
        )
        out_flow = a2a_resuslts[0]
        print(f"out_flow.shape: {out_flow.shape}")
        nodes_results = out_flow.chunk(2)
        print(f"layer id: {self.expert_key} after a2a outflow: {out_flow}")
        input()

        if self.locality_aware:
            route_indices, loads, score = brt_dist.batched_exchange(
                [route_indices, loads, score], a2a_resuslts[2]
            )
            print(f"reorder indices {a2a_resuslts[2]}")
            brt_dist.set_reorder_indices(a2a_resuslts[2])

        out_flow.route_indices = route_indices
        out_flow.in_loads = loads
        out_flow.out_loads = a2a_resuslts[1]
        out_flow.score = score

        return out_flow


@register_fabric("distributed_fused_combine")
class DistributedFusedCombineFabric(FusedCombineFabric):
    def __init__(
        self,
        flow_num,
        sparse,
        reduction,
        granularity_padding,
        locality_aware: bool = False,
    ) -> None:
        assert granularity_padding == False
        self.locality_aware = locality_aware
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=sparse,
            granularity_padding=False,
        )
        self.transform = True

    def forward(
        self,
        in_flow: torch.Tensor,
    ) -> List[torch.Tensor]:
        route_indices = in_flow.route_indices
        in_loads = in_flow.in_loads
        out_loads = in_flow.out_loads
        score = in_flow.score
        in_flow = brt_dist.size_known_group_asymmetry_all_to_all(
            in_flow, out_loads, in_loads
        )
        if self.transform:
            out_flow = combine_with_src_indices(
                in_flow, route_indices, in_loads, auto_pad=True, gates=score
            )
        else:
            out_flow = combine_with_src_indices(in_flow, route_indices, in_loads, None)

        if self.locality_aware:
            reorder_indices = brt_dist.get_reorder_indices()
            print(f"reorder indices {reorder_indices}")
            out_flow = brt_dist.exchange(out_flow, reorder_indices)
            brt_dist.set_reorder_indices(None)

        return out_flow
