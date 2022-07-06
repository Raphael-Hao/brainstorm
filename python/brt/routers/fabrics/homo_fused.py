from typing import List, Tuple

import torch
from brt._C.router import (
    generate_local_indices,
    route_back_with_local_indices,
    route_with_local_indices,
)
from brt.common import log

from ..proto_tensor import (
    ProtoTensor,
    init_proto_tensor,
    make_proto_tensor_cls,
    to_torch_tensor,
)
from .fabric import FabricFactory
from .generic import CombineSF, DispatchSF

logger = log.get_logger(__file__)

__all__ = [
    "HomoFusedDispatchSF",
    "HomoFusedCombineSF",
]


_homo_proto_tensor_created = False


def check_homo_proto_tensor_clos():
    from brt.routers import ProtoTensor

    assert hasattr(ProtoTensor, "score")
    assert hasattr(ProtoTensor, "score_stack")


def once_make_homo_proto_tensor_cls():
    global _homo_proto_tensor_created
    if _homo_proto_tensor_created is False:
        make_proto_tensor_cls(["score"], [None])
        _homo_proto_tensor_created = True
    check_homo_proto_tensor_clos()


@FabricFactory.register("homo_fused_dispatch")
class HomoFusedDispatchSF(DispatchSF):
    def __init__(self, path_num, **kwargs):
        super().__init__(path_num, **kwargs)
        once_make_homo_proto_tensor_cls()
        supported_capacities = kwargs.get("supported_capacities", None)
        if supported_capacities is None:
            self.supported_capacities = supported_capacities
        else:
            self.supported_capacities = torch.tensor(
                supported_capacities, dtype=torch.int32
            )
        self.post_transform = kwargs.get("post_transform", False)
        if self.post_transform and self.transform:
            logger.warning(
                "Post_transform and transform are both True, double transform with same score!"
            )

    def forward(
        self, in_flow: ProtoTensor, hot_mask: torch.Tensor, score: torch.Tensor
    ) -> ProtoTensor:
        local_indices, loads = self.gen_indices(hot_mask)
        in_flow = self.pack_invalid_flow(in_flow)
        (
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
        ) = to_torch_tensor(in_flow, copy_stack=True)

        if self.transform:
            out_flow_data = route_with_local_indices(
                in_flow_data, local_indices, loads, score
            )
        else:
            out_flow_data = route_with_local_indices(
                in_flow_data, local_indices, loads, None
            )
        out_flow = init_proto_tensor(
            out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack_dict
        )
        if self.post_transform:
            out_flow.pack(local_indices, loads, score=score)
        else:
            out_flow.pack(local_indices, loads)

        return out_flow

    def gen_indices(self, hot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.supported_capacities is not None:
            self.supported_capacities = self.supported_capacities.to(hot_mask.device)

        local_indices, loads = generate_local_indices(
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

@FabricFactory.register("homo_fused_combine")
class HomoFusedCombineSF(CombineSF):
    def __init__(self, path_num, **kwargs) -> None:
        super().__init__(path_num, **kwargs)
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
        ) = to_torch_tensor(in_flow, copy_stack=True)

        local_indices = in_flow_tag_stack.pop()
        loads = in_flow_load_stack.pop()
        for extra_attr_key in extra_attr_stack.keys():
            if extra_attr_key == "score_stack":
                score = extra_attr_stack["score_stack"].pop()
            else:
                extra_attr_stack[extra_attr_key].pop()

        # self.start_timer()
        if self.transform:
            out_flow_data = route_back_with_local_indices(
                in_flow_data, local_indices, loads, score
            )
        else:
            out_flow_data = route_back_with_local_indices(
                in_flow_data, local_indices, loads, None
            )

        # self.end_timer("route_back_with_local_indices")

        out_flow = init_proto_tensor(
            out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack
        )

        return out_flow
