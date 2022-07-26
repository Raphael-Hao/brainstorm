from typing import List, Tuple, Union

import torch
from brt.router.utils import generate_dst_indices
from brt._C.router import route_back_with_dst_indices, route_with_dst_indices
from brt.runtime import log
from brt.router.fabric.base import register_fabric
from brt.router.fabric.generic import CombineFabric, DispatchFabric
from brt.router.proto_tensor import (
    ProtoTensor,
    init_proto_tensor,
    make_proto_tensor_cls,
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
        throttling=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
    ):
        super().__init__(
            throttling=throttling, route_logic=route_logic, transform=transform
        )

    def forward(
        self,
        in_flow: ProtoTensor,
        local_indices: torch.Tensor,
        loads: torch.Tensor,
        score: torch.Tensor,
    ) -> ProtoTensor:
        in_flow = self.pack_invalid_flow(in_flow)
        (
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
        ) = to_torch_tensor(in_flow, copy_stack=True, shallow=True)

        if self.transform:
            out_flow_data = route_with_dst_indices(
                in_flow_data, local_indices, loads, score
            )
        else:
            out_flow_data = route_with_dst_indices(
                in_flow_data, local_indices, loads, None
            )
        out_flow = init_proto_tensor(
            out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack_dict
        )

        out_flow.pack(local_indices, loads)

        return out_flow

    def gen_indices(self, hot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.supported_capacities is not None:
            self.supported_capacities = self.supported_capacities.to(hot_mask.device)

        local_indices, loads = generate_dst_indices(
            hot_mask.to(torch.int32), self.supported_capacities
        )
        # self.end_timer("generate_local_indices")

        return local_indices, loads

    def pack_invalid_flow(self, in_flow):

        from ..proto_tensor import ProtoTensor  # we need to keep ProtoTensor updated

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
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        check_homo_proto_tensor_clos()
        supported_capacities = kwargs.get("supported_capacities", None)
        if supported_capacities is None:
            self.supported_capacities = supported_capacities
        else:
            self.supported_capacities = torch.tensor(
                supported_capacities, dtype=torch.int32
            )
        self.transform = kwargs.get("transform", False)

    def forward(self, in_flow: ProtoTensor) -> ProtoTensor:
        (
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack,
        ) = to_torch_tensor(in_flow, copy_stack=True, shallow=True)

        local_indices = in_flow_tag_stack.pop()
        loads = in_flow_load_stack.pop()
        for extra_attr_key in extra_attr_stack.keys():
            if extra_attr_key == "score_stack":
                score = extra_attr_stack["score_stack"].pop()
            else:
                extra_attr_stack[extra_attr_key].pop()

        # self.start_timer()
        if self.transform:
            out_flow_data = route_back_with_dst_indices(
                in_flow_data, local_indices, loads, score
            )
        else:
            out_flow_data = route_back_with_dst_indices(
                in_flow_data, local_indices, loads, None
            )

        # self.end_timer("route_back_with_local_indices")

        out_flow = init_proto_tensor(
            out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack
        )

        return out_flow
