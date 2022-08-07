from typing import List, Tuple, Union

import torch
from brt.router.utils import generate_dst_indices
from brt._C.router import (
    dispatch_with_dst_indices_1d,
    dispatch_with_dst_indices_2d,
    combine_with_src_indices,
)
from brt.runtime import log
from brt.router.fabric.base import register_fabric
from brt.router.fabric.generic import CombineFabric, DispatchFabric
from brt.runtime.proto_tensor import (
    ProtoTensor,
    init_proto_tensor,
    to_torch_tensor,
)

logger = log.get_logger(__file__)

__all__ = [
    "HomoFusedDispatchFabric",
    "HomoFusedCombineFabric",
]


@register_fabric("homo_fused_dispatch")
class HomoFusedDispatchFabric(DispatchFabric):
    def __init__(
        self,
        capacity_padding=False,
        throttling=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
    ):
        self.capacity_padding = capacity_padding
        super().__init__(
            throttling=throttling, route_logic=route_logic, transform=transform
        )

    def dispatch(
        self,
        in_flows: List[ProtoTensor],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[ProtoTensor]:
        all_out_flows = []

        for flow_idx, flow in enumerate(in_flows):
            (
                in_flow_data,
                in_flow_tag_stack,
                in_flow_load_stack,
                extra_attr_stack_dict,
            ) = to_torch_tensor(flow, return_stack=True)

            if self.route_logics[flow_idx] == "1d":
                if self.transforms[flow_idx]:
                    out_flow_data = dispatch_with_dst_indices_1d(
                        in_flow_data, route_indices, loads, self.capacity_padding, score
                    )
                else:
                    out_flow_data = dispatch_with_dst_indices_1d(
                        in_flow_data, route_indices, loads, self.capacity_padding
                    )
                out_flow = init_proto_tensor(
                    out_flow_data,
                    in_flow_tag_stack,
                    in_flow_load_stack,
                    extra_attr_stack_dict,
                )
                out_flow.pack(route_indices, loads)
            elif self.route_logics[flow_idx] == "2d":
                in_flow_data = in_flow_data.transpose(0, 1).contiguous()
                out_flow_data = dispatch_with_dst_indices_2d(
                    in_flow_data, route_indices, loads, self.capacity_padding
                )
                out_flow = init_proto_tensor(
                    out_flow_data,
                    in_flow_tag_stack,
                    in_flow_load_stack,
                    extra_attr_stack_dict,
                )
                out_flow.pack(route_indices, loads)
            else:
                raise ValueError("route_logic must be 1d or 2d")
            all_out_flows.append(out_flow)

        return all_out_flows

    def pack_invalid_flow(self, in_flow):

        from ...runtime.proto_tensor import (
            ProtoTensor,
        )  # we need to keep ProtoTensor updated

        if isinstance(in_flow, ProtoTensor):
            pass

        elif isinstance(in_flow, torch.Tensor):
            in_flow = init_proto_tensor(in_flow)

        elif isinstance(in_flow, (List, Tuple)):
            in_flow = type(in_flow)([self.pack_invalid_flow(f) for f in in_flow])

        return in_flow

    def remove_needless_pack(self, out_flow):
        return out_flow


@register_fabric("homo_fused_combine")
class HomoFusedCombineFabric(CombineFabric):
    def __init__(self, flow_num, sparse, reduction) -> None:
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=sparse,
            granularity_padding=False,
        )

    def dispatch(self, in_flows: List[ProtoTensor]) -> List[ProtoTensor]:
        out_flows = []
        for flow_idx in self.flow_num:
            in_flow = in_flows[flow_idx]
            (
                in_flow_data,
                in_flow_tag_stack,
                in_flow_load_stack,
                extra_attr_stack,
            ) = to_torch_tensor(in_flow, return_stack=True)

            local_indices = in_flow_tag_stack.pop()
            loads = in_flow_load_stack.pop()
            for extra_attr_key in extra_attr_stack.keys():
                if extra_attr_key == "score_stack":
                    score = extra_attr_stack["score_stack"].pop()
                else:
                    extra_attr_stack[extra_attr_key].pop()

            # self.start_timer()
            if self.transform:
                out_flow_data = combine_with_src_indices(
                    in_flow_data, local_indices, loads, score
                )
            else:
                out_flow_data = combine_with_src_indices(
                    in_flow_data, local_indices, loads, None
                )

            # self.end_timer("route_back_with_local_indices")

            out_flow = init_proto_tensor(
                out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack
            )
            out_flows.append(out_flow)

        return out_flows
