# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, List, Dict, Any

import torch
from brt.router.utils import assert_compatibility
from brt.router.fabric.generic import DispatchFabric, CombineFabric
from brt.router.fabric.base import register_fabric
from brt.runtime.grid_tensor import GridTensor


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
        """ identity dispatch fabric
        path_num: number of paths
        flow_num: number of input flows
        """
        self.check_compatibility(kwargs)
        super().__init__(
            flow_num=flow_num,
            route_logic=route_logic,
            transform=transform,
            index_format=None,
            **kwargs
        )
        self.path_num = path_num

    def forward(
        self,
        in_flow: Union[GridTensor, List[GridTensor]],
        hot_mask: torch.Tensor,
        runtime_capacities: torch.Tensor = None,
        score: torch.Tensor = None,
    ) -> Union[List[GridTensor], List[List[GridTensor]]]:

        if self.flow_num == 1:
            in_flows = [in_flow]
        else:
            in_flows = in_flow

        all_out_flows = self.dispatch(in_flows, None, None, None)

        if self.flow_num == 1:
            all_out_flows = all_out_flows[0]
        return all_out_flows

    def dispatch(
        self,
        in_flows: List[GridTensor],
        hot_mask: torch.Tensor,
        runtime_capacities: torch.Tensor,
        score: torch.Tensor,
    ) -> List[List[GridTensor]]:
        all_out_flows = []
        for flow_idx in range(self.flow_num):
            flow = in_flows[flow_idx]
            if self.route_logics[flow_idx] == "1d":
                route_shape = list(flow.shape[1:])
            elif self.route_logics[flow_idx] == "2d":
                route_shape = list(flow.shape[1:])
                route_shape[0] = 1
                flow = flow.transpose(0, 1).contiguous()
            else:
                raise ValueError("route_logic must be 1d or 2d")
            out_flows = []
            for path_id in range(self.path_num):
                if self.route_logics[flow_idx] == "1d":
                    dispatched_flow = flow
                    if self.transforms[flow_idx]:
                        route_shape = list(flow.shape[1:])
                        dispatched_flow = dispatched_flow * score[:, path_id].view(
                            (-1,) + (1,) * len(route_shape)
                        )
                elif self.route_logics[flow_idx] == "2d":
                    dispatched_flow = flow[path_id].contiguous()
                else:
                    raise ValueError("route_logic must be 1d or 2d")

                out_flow = dispatched_flow
                out_flows.append(out_flow)
            all_out_flows.append(out_flows)
        return all_out_flows


@register_fabric("identity_combine")
class IdentityCombineFabric(CombineFabric):
    def __init__(self, flow_num: int, **kwargs):
        self.check_compatibility(kwargs)
        super().__init__(flow_num=flow_num, reduction="add", **kwargs)

    def forward(
        self, in_flows: Union[List[GridTensor], List[List[GridTensor]]]
    ) -> Union[GridTensor, List[GridTensor]]:
        if self.flow_num == 1:
            in_flows = [in_flows]

        out_flows = self.combine(in_flows)

        if self.flow_num == 1:
            return out_flows[0]

        return out_flows

    def combine(self, in_flows: List[List[GridTensor]]) -> List[GridTensor]:
        out_flows = []

        for flow_idx in range(self.flow_num):
            original_shape = in_flows[flow_idx][0].shape
            out_flow = (
                torch.cat(in_flows[flow_idx], dim=0)
                .reshape(-1, 1, *original_shape[1:])
                .sum(dim=0)
            )
            out_flows.append(out_flow)

        return out_flows

    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        sparse = kwargs.pop("sparse", True)
        assert_compatibility(self, "sparse", True, sparse)
        reduction = kwargs.pop("reduction", "add")
        assert_compatibility(self, "reduction", "add", reduction)
