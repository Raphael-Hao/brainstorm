from typing import Any, Dict, List, Tuple, Union

import torch
from brt.cpp.router import (
    generate_local_indices,
    route_back_with_local_indices,
    route_with_local_indices,
)
from brt.primitive import router
from brt.routers.base import GatherRouter, ScatterRouter
from brt.routers.proto_tensor import (
    ProtoTensor,
    init_proto_tensor,
    make_proto_tensor_cls,
    to_torch_tensor,
)
from brt.routers.symbolic import symbolic_gather_route, symbolic_scatter_route

__all__ = [
    "make_homo_proto_tensor_cls",
    "HomoFusedScatterRouter",
    "HomoFusedGatherRouter",
]


def make_homo_proto_tensor_cls():
    make_proto_tensor_cls(["gates"], [None])
    from brt.routers import ProtoTensor

    assert hasattr(ProtoTensor, "gates")
    assert hasattr(ProtoTensor, "gates_stack")


@router
class HomoFusedScatterRouter(ScatterRouter):
    def __init__(
        self,
        dst_num: int,
        route_func,
        route_method: str = "topk",
        residual_dst: int = -1,
        transform: bool = True,
        supported_capacities: List[int] = None,
        **kwargs,
    ):
        super().__init__(
            dst_num, route_func, route_method, residual_dst, transform, **kwargs
        )
        if supported_capacities is None:
            self.supported_capacities = supported_capacities
        else:
            self.supported_capacities = torch.tensor(
                supported_capacities, dtype=torch.int64
            )

    def dispatch(
        self,
        in_flow_data: torch.Tensor,
        in_flow_tag_stack: List[torch.Tensor],
        in_flow_load_stack: List[int],
        extra_attr_stack: Dict[str, List[Any]],
        route_indices: torch.Tensor,
        gates: torch.Tensor,
    ) -> ProtoTensor:
        # currently we assume that the ProtoTensor is dense one
        _in_flow_tag = in_flow_tag_stack.pop()
        _in_flow_load = in_flow_load_stack.pop()

        for key in extra_attr_stack.keys():
            extra_attr_stack[key].pop()
        if self.supported_capacities is not None:
            self.supported_capacities = self.supported_capacities.to(
                in_flow_data.device
            )
        self.start_timer()
        local_indices, loads = generate_local_indices(
            route_indices, self.supported_capacities
        )
        self.end_timer("generate_local_indices")

        self.start_timer()
        if self.transform:
            out_flow_data = route_with_local_indices(
                in_flow_data, local_indices, loads, gates
            )
            gates = None
        else:
            out_flow_data = route_with_local_indices(
                in_flow_data, local_indices, loads, None
            )
        self.end_timer("route_with_local_indices")
        out_flow = init_proto_tensor(
            out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack
        )
        out_flow.pack(local_indices, loads, gates=gates)

        return out_flow

    def remove_needless_pack(self, out_flow):
        return out_flow

    def symbolic_route(self, inputs: torch.Tensor) -> torch.Tensor:
        return symbolic_scatter_route(inputs, 1)[0]


@router
class HomoFusedGatherRouter(GatherRouter):
    def __init__(
        self, dst_num: int, reduction: str = "add", sparse=True, transform=False
    ):
        super().__init__(
            dst_num=dst_num, reduction=reduction, sparse=sparse, transform=transform
        )
        self.sparse = sparse
        self.reduction = reduction

    def pack_invalid_flow(self, in_flow):
        # assert in_flow.tag
        return in_flow

    def combine(self, in_flow: ProtoTensor) -> Union[ProtoTensor, torch.Tensor]:

        (
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack,
        ) = to_torch_tensor(in_flow, copy_stack=True)

        local_indices = in_flow_tag_stack.pop()
        loads = in_flow_load_stack.pop()
        gates = extra_attr_stack["gates_stack"].pop()

        self.start_timer()
        if self.transform:
            out_flow_data = route_back_with_local_indices(
                in_flow_data, local_indices, loads, gates
            )
        else:
            out_flow_data = route_back_with_local_indices(
                in_flow_data, local_indices, loads, None
            )
        self.end_timer("route_back_with_local_indices")

        out_flow = init_proto_tensor(
            out_flow_data, in_flow_tag_stack, in_flow_load_stack, extra_attr_stack
        )

        return out_flow

    def symbolic_route(self, inputs: torch.Tensor) -> torch.Tensor:
        return symbolic_gather_route([inputs], 1)
