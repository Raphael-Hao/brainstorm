from typing import List, Tuple, Union

import brt._C.router as c_router
import torch
from brt.router.fabric.base import register_fabric
from brt.router.fabric.generic import CombineFabric, DispatchFabric
from brt.runtime import log
from brt.runtime.grid_tensor import GridTensor, init_grid_tensor, to_torch_tensor

logger = log.get_logger(__file__)

__all__ = [
    "FusedDispatchFabric",
    "FusedCombineFabric",
]


@register_fabric("fused_dispatch")
class FusedDispatchFabric(DispatchFabric):
    def __init__(
        self,
        flow_num: int,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        **kwargs
    ):
        kwargs["index_format"] = "seat_index"
        super().__init__(
            flow_num, route_logic=route_logic, transform=transform, **kwargs
        )

    def dispatch(
        self,
        in_flows: List[GridTensor],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> Tuple[List[GridTensor], torch.Tensor]:
        all_out_flows = []
        for flow_idx, flow in enumerate(in_flows):
            (
                flow_data,
                flow_tag_stack,
                flow_load_stack,
                extra_attr_dict,
            ) = to_torch_tensor(flow, retrieve_attr=True)

            flow_tag = flow_tag_stack[-1]
            _flow_load = flow_load_stack[-1]
            assert flow_tag == None

            flow_tag_stack = flow_tag_stack[:-1]
            flow_load_stack = flow_load_stack[:-1]
            flow_extra_attr_dict = extra_attr_dict

            routed_data = c_router.dispatch_with_indices_and_loads(
                flow_data,
                route_indices,
                loads,
                score,
                self.is_tag_index,
                flow_tag,
                self.max_path_padding,
                self.max_path_load,
                self.route_logics[flow_idx] == "1d",
            )
            out_flow = init_grid_tensor(
                routed_data,
                flow_tag_stack + [route_indices],
                flow_load_stack + [loads],
                flow_extra_attr_dict,
            )
            all_out_flows.append(out_flow)

        return all_out_flows, score


@register_fabric("fused_combine")
class FusedCombineFabric(CombineFabric):
    def __init__(self, flow_num, reduction, transform, **kwargs) -> None:
        kwargs["index_format"] = "seat_index"
        super().__init__(
            flow_num=flow_num, reduction=reduction, transform=transform, **kwargs
        )

    def combine(
        self,
        in_flows: List[GridTensor],
        residual_flows: List[GridTensor],
        score: torch.Tensor,
    ) -> List[GridTensor]:
        out_flows = []
        for flow_idx, in_flow in enumerate(in_flows):
            (
                in_flow_data,
                in_flow_tag_stack,
                in_flow_load_stack,
                extra_attr_stack,
            ) = to_torch_tensor(in_flow, retrieve_attr=True)

            residual_flow = (
                residual_flows[flow_idx] if residual_flows is not None else None
            )

            route_indices = in_flow_tag_stack.pop()
            in_flows_load = in_flow_load_stack.pop()

            out_flow_data = c_router.combine_with_indices_and_loads(
                in_flow_data,
                route_indices,
                in_flows_load,
                score,
                residual_flow,
                max_path_padding=self.max_path_padding,
                ever_padded=self.ever_padded,
            )

            out_flow = init_grid_tensor(
                out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack
            )
            out_flows.append(out_flow)

        return out_flows
