# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, List, Tuple

import torch
from brt.router.fabric.generic import DispatchFabric, CombineFabric
from brt.router.fabric.base import register_fabric


@register_fabric("single_cell_dispatch")
class SingleCellDispatchFabric(DispatchFabric):
    def __init__(
        self,
        flow_num: int,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        **kwargs
    ):
        super().__init__(
            flow_num=flow_num, route_logic=route_logic, transform=transform, **kwargs
        )

    def forward(
        self,
        in_flow: Union[torch.Tensor, List[torch.Tensor]],
        hot_mask: torch.Tensor,
        runtime_capacities: torch.Tensor = None,
        score: torch.Tensor = None,
    ) -> Tuple[Union[List[torch.Tensor], List[List[torch.Tensor]]], torch.Tensor]:

        if self.flow_num == 1:
            in_flows = [in_flow]
        else:
            in_flows = in_flow
        if (hot_mask.numel() == 0):
            loads_shape = [hot_mask.size(1)]
            loads = torch.zeros(loads_shape, dtype=hot_mask.dtype, device=hot_mask.device)
        else:
            loads = hot_mask.view(-1)
        self.capture_flow_stats(in_flows, hot_mask, loads)

        all_out_flows, score = self.dispatch(in_flows, hot_mask, loads, score)

        if self.flow_num == 1:
            all_out_flows = all_out_flows[0]

        return all_out_flows, score

    def dispatch(
        self,
        in_flows: List[torch.Tensor],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[List[torch.Tensor]]:
        all_out_flows = []
        for flow_idx, flow in enumerate(in_flows):
            flow_data =flow

            if self.route_logics[flow_idx] == "1d":
                route_shape = list(flow_data.shape[1:])
            elif self.route_logics[flow_idx] == "2d":
                route_shape = list(flow_data.shape[1:])
                route_shape[0] = 1
                flow_data = flow_data.transpose(0, 1).contiguous()
            else:
                raise ValueError("route_logic must be 1d or 2d")
            out_flows = []
            cpu_loads = loads.cpu()
            for path_id, load in enumerate(cpu_loads):
                if load == 0:
                    out_flow = torch.zeros(
                            0,
                            *route_shape,
                            dtype=flow_data.dtype,
                            device=flow_data.device,
                        )
                else:
                    if self.route_logics[flow_idx] == "1d":
                        dispatched_data = flow_data
                    elif self.route_logics[flow_idx] == "2d":
                        dispatched_data = flow_data[path_id : path_id + 1].contiguous()
                    else:
                        raise ValueError("route_logic must be 1d or 2d")

                    if self.transforms[flow_idx]:
                        dispatched_data = dispatched_data * score[:, path_id].view(
                            (-1,) + (1,) * len(route_shape)
                        )
                    out_flow =dispatched_data

                out_flows.append(out_flow)
            all_out_flows.append(out_flows)
        return all_out_flows, score


@register_fabric("single_cell_combine")
class SingleCellCombineFabric(CombineFabric):
    def __init__(self, flow_num: int, reduction="add", transform=False, **kwargs):
        super().__init__(
            flow_num=flow_num, reduction=reduction, transform=transform, **kwargs
        )
        assert reduction == "add"
        assert not transform

    def combine(
        self,
        in_flows: List[List[torch.Tensor]],
        residual_flows: List[torch.Tensor],
        score: torch.Tensor,
    ) -> List[torch.Tensor]:
        out_flows = []

        for flow_idx, in_flow in enumerate(in_flows):
            is_residual = residual_flows is not None
            residual_flow = (
                residual_flows[flow_idx] if residual_flows is not None else None
            )
            in_flows_data = in_flow
            out_flow_data = torch.cat(in_flows_data, dim=0)

            if out_flow_data.numel() > 0:
                out_flow_data = out_flow_data.sum(dim=0, keepdim=True)
                out_flow = out_flow_data
            else:
                if is_residual:
                    out_flow = residual_flow
                else:
                    out_flow = out_flow_data
            out_flows.append(out_flow)
        return out_flows
