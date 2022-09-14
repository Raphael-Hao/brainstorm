# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, List, Dict, Any

import torch
from brt.router.utils import assert_compatibility
from brt.router.fabric.generic import DispatchFabric, CombineFabric
from brt.router.fabric.base import register_fabric
from brt.runtime.proto_tensor import ProtoTensor, init_proto_tensor


@register_fabric("identity_dispatch")
class IndentityDispatchFabric(DispatchFabric):
    def __init__(
        self,
        path_num: int,
        flow_num: int,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        **kwargs
    ):
        self.check_compatibility(kwargs)
        super().__init__(
            flow_num=flow_num,
            route_logic=route_logic,
            transform=transform,
            index_format=None,
            **kwargs
        )
        self.path_num = path_num

    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        index_format = kwargs.pop("index_format", None)
        assert_compatibility(self, "index_format", None, index_format)

    def forward(
        self,
        in_flow: Union[torch.Tensor, List[torch.Tensor]],
        route_indices: torch.Tensor,
        real_loads: torch.Tensor,
        score: torch.Tensor = None,
    ) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        if self.flow_num == 1:
            in_flows = [in_flow]
        else:
            in_flows = in_flow
        all_out_flows = self.dispatch(in_flows, route_indices, real_loads, score)
        if self.flow_num == 1:
            return all_out_flows[0]
        return all_out_flows

    def dispatch(
        self,
        in_flows: List[torch.Tensor],
        route_indices: torch.Tensor,
        real_loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[List[torch.Tensor]]:
        all_out_flows = []
        for flow_idx in range(self.flow_num):
            flow = in_flows[flow_idx]
            if self.route_logics[flow_idx] == "1d":
                route_shape = list(flow.shape[1:])
            elif self.route_logics[flow_idx] == "2d":
                route_shape = list(flow.shape[1:])
                route_shape[0] = 1
            else:
                raise ValueError("route_logic must be 1d or 2d")
            out_flows = []
            for path_id in range(self.path_num):

                if self.route_logics[flow_idx] == "1d":
                    dispatched_flow = flow
                elif self.route_logics[flow_idx] == "2d":
                    dispatched_flow = flow[:, path_id : path_id + 1].contiguous()
                else:
                    raise ValueError("route_logic must be 1d or 2d")

                if self.transforms[flow_idx]:
                    out_flow = dispatched_flow * score[:, path_id].view(
                        (-1,) + (1,) * len(route_shape)
                    )
                else:
                    out_flow = dispatched_flow
                out_flows.append(out_flow)
            all_out_flows.append(out_flows)
        return all_out_flows


@register_fabric("identity_combine")
class IdentityCombineFabric(CombineFabric):
    def __init__(self, flow_num: int, granularity_padding=False, **kwargs):
        self.check_compatibility(kwargs)
        super().__init__(
            flow_num=flow_num,
            reduction="add",
            sparse=True,
            granularity_padding=granularity_padding,
            **kwargs
        )

    def forward(
        self, in_flows: Union[List[ProtoTensor], List[List[ProtoTensor]]]
    ) -> Union[ProtoTensor, List[ProtoTensor]]:
        if self.flow_num == 1:
            in_flows = [in_flows]

        out_flows = self.combine(in_flows)

        if self.flow_num == 1:
            return out_flows[0]

        return out_flows

    def combine(self, in_flows: List[List[ProtoTensor]]) -> List[ProtoTensor]:
        out_flows = []

        for flow_idx in range(self.flow_num):
            out_flow = torch.stack(in_flows[flow_idx], dim=0).sum(dim=0)
            out_flows.append(out_flow)

        return out_flows

    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        sparse = kwargs.pop("sparse", True)
        assert_compatibility(self, "sparse", True, sparse)
        reduction = kwargs.pop("reduction", "add")
        assert_compatibility(self, "reduction", "add", reduction)
