# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Tuple, Union

import brt._C.router as c_router
import brt.runtime.distributed as brt_dist
import torch
import torch.distributed as dist
from brt.router.fabric.base import register_fabric
from brt.router.fabric.fused import FusedCombineFabric, FusedDispatchFabric
from brt.runtime.grid_tensor import GridTensor, init_grid_tensor, to_torch_tensor


@register_fabric("distributed_fused_dispatch")
class DistributedFusedDispatchFabric(FusedDispatchFabric):
    def __init__(
        self,
        flow_num: int,
        capacity_padding=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        task_locality: bool = False,
        **kwargs,
    ):
        self.task_locality = task_locality
        kwargs["index_format"] = "seat_index"
        super().__init__(
            flow_num=flow_num,
            capacity_padding=capacity_padding,
            route_logic=route_logic,
            transform=transform,
            **kwargs,
        )

    def dispatch(
        self,
        in_flows: List[GridTensor],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> Tuple[List[GridTensor], torch.Tensor]:
        all_out_flows = []
        for flow_idx in range(self.flow_num):
            in_flow = in_flows[flow_idx]
            (
                flow_data,
                flow_tag_stack,
                flow_load_stack,
                extra_attr_dict,
            ) = to_torch_tensor(in_flow, retrieve_attr=True)
            routed_data = c_router.dispatch_with_indices_and_loads(
                flow_data,
                route_indices,
                loads,
                score if self.transforms[flow_idx] else None,
                False,
                None,
                self.max_path_padding,
                self.max_path_load,
                self.route_logics[flow_idx] == "1d",
            )

            out_flow_data, out_loads, in_loads = brt_dist.group_sparse_a2a(
                routed_data[0], loads, self.max_path_load, self.max_path_load
            )
            if self.task_locality:
                world_size = dist.get_world_size()
                world_rank = dist.get_rank()
                num_local_tasks = out_loads.size(0) // world_size
                task_ids = torch.empty(
                    out_flow_data.size(0),
                    dtype=torch.int64,
                    device=out_flow_data.device,
                )
                base_idx = 0
                for i in range(num_local_tasks):
                    task_total_load = (
                        out_loads[i * world_size : (i + 1) * world_size].sum().item()
                    )
                    task_ids[base_idx : base_idx + task_total_load].fill_(
                        world_rank * num_local_tasks + i
                    )
                    base_idx += task_total_load
                score = task_ids

            out_flow = init_grid_tensor(
                out_flow_data, flow_tag_stack, flow_load_stack, extra_attr_dict
            )
            out_flow.pack(
                route_indices,
                out_loads,
                in_loads=in_loads,
            )
            all_out_flows.append(out_flow)

        return all_out_flows, score


@register_fabric("distributed_fused_combine")
class DistributedFusedCombineFabric(FusedCombineFabric):
    def __init__(
        self,
        flow_num,
        reduction="add",
        transform=False,
        task_locality: bool = False,
        **kwargs,
    ) -> None:
        self.task_locality = task_locality
        kwargs["index_format"] = "seat_index"
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            transform=transform,
            **kwargs,
        )

    def combine(
        self,
        in_flows: List[GridTensor],
        residual_flows: List[GridTensor],
        score: torch.Tensor,
    ) -> List[GridTensor]:
        out_flows = []
        for idx, in_flow in enumerate(in_flows):
            in_flow, route_indices, in_loads, pre_attr_dict = in_flow.unpack("in_loads")
            out_loads = pre_attr_dict["in_loads"]
            (
                in_flow_data,
                in_flow_tag_stack,
                in_flow_load_stack,
                extra_attr_stack,
            ) = to_torch_tensor(in_flow, retrieve_attr=True)

            residual_flow = residual_flows[idx] if residual_flows is not None else None

            in_flow_data = brt_dist.size_known_group_sparse_a2a(
                in_flow_data, in_loads, out_loads, self.max_path_padding
            )
            out_flow_data = c_router.combine_with_indices_and_loads(
                in_flow_data,
                route_indices,
                out_loads,
                score,
                residual_flow,
                max_path_padding=self.max_path_padding,
                ever_padded=self.ever_padded,
            )
            out_flow = init_grid_tensor(
                out_flow_data[0],
                in_flow_tag_stack,
                in_flow_load_stack,
                extra_attr_stack,
            )
            out_flows.append(out_flow)

        return out_flow, score
