# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, List, Tuple

import torch
from brt.router.fabric.generic import DispatchFabric, CombineFabric
from brt.router.fabric.base import register_fabric
from brt.runtime.grid_tensor import GridTensor, init_grid_tensor, to_torch_tensor


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
        in_flow: Union[GridTensor, List[GridTensor]],
        hot_mask: torch.Tensor,
        runtime_capacities: torch.Tensor = None,
        score: torch.Tensor = None,
    ) -> Tuple[Union[List[GridTensor], List[List[GridTensor]]], torch.Tensor]:

        if self.flow_num == 1:
            in_flows = [in_flow]
        else:
            in_flows = in_flow
        loads = hot_mask.view(-1)
        self.capture_flow_stats(in_flows, hot_mask, loads)

        all_out_flows, score = self.dispatch(in_flows, hot_mask, loads, score)

        if self.flow_num == 1:
            all_out_flows = all_out_flows[0]

        return all_out_flows, score

    def dispatch(
        self,
        in_flows: List[GridTensor],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[List[GridTensor]]:
        all_out_flows = []
        for flow_idx, flow in in_flows:
            (flow_data, tag_stack, load_stack, extra_attr_dict,) = to_torch_tensor(
                flow, retrieve_attr=True
            )

            tag_stack = tag_stack[:-1]
            load_stack = load_stack[:-1]

            if self.route_logics[flow_idx] == "1d":
                route_shape = list(flow_data.shape[1:])
            elif self.route_logics[flow_idx] == "2d":
                route_shape = list(flow_data.shape[1:])
                route_shape[0] = 1
                flow_data = flow_data.transpose(0, 1).contiguous()
            else:
                raise ValueError("route_logic must be 1d or 2d")
            out_flows = []
            cpu_loads = loads.cpu().numpy()
            for path_id, load in enumerate(cpu_loads):
                if load == 0:
                    out_flow = init_grid_tensor(
                        torch.zeros(
                            0,
                            *route_shape,
                            dtype=flow_data.dtype,
                            device=flow_data.device,
                        ),
                        tag_stack,
                        load_stack,
                        extra_attr_dict,
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
                    out_flow = init_grid_tensor(
                        dispatched_data, tag_stack, load_stack, extra_attr_dict,
                    )
                out_flow.pack(route_indices[:, path_id], loads[path_id])
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
        in_flows: List[List[GridTensor]],
        residual_flows: List[GridTensor],
        score: torch.Tensor,
    ) -> List[GridTensor]:
        out_flows = []

        for flow_idx, in_flow in enumerate(in_flows):
            residual_flow = (
                residual_flows[flow_idx] if residual_flows is not None else None
            )
            in_flows_data = []
            out_flow_tag_stack = None
            out_flow_load_stack = None

            for flow in in_flow:
                (
                    data,
                    out_flow_tag_stack,
                    out_flow_load_stack,
                    extra_attr_dict,
                ) = to_torch_tensor(flow, retrieve_attr=True)
                in_flows_data.append(data)
            out_flow_tag_stack = out_flow_tag_stack[:-1]
            out_flow_load_stack = out_flow_load_stack[:-1]

            out_flow_data = torch.cat(in_flows_data, dim=0)

            if out_flow_data.numel() > 0:
                out_flow_data = out_flow_data.sum(dim=0, keepdim=True)
                out_flow = init_grid_tensor(
                    out_flow_data,
                    out_flow_tag_stack,
                    out_flow_load_stack,
                    extra_attr_dict,
                )
                new_tag = torch.tensor((1,), dtype=torch.int32, device=out_flow_data.device)
                new_load = torch.tensor((1,), dtype=torch.int32, device=out_flow_data.device)
                out_flow.pack(new_tag, new_load)
            else:
                out_flow = residual_flow
            out_flows.append(out_flow)
        return out_flows
