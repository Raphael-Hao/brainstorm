from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from brt.routers.base import GatherRouter, ScatterRouter
from brt.routers.proto_tensor import (
    ProtoTensor,
    init_proto_tensor,
    make_proto_tensor_cls,
    to_torch_tensor,
)
from brt.routers.symbolic import symbolic_gather_route, symbolic_scatter_route
from brt.cpp.router import generate_indices_with_load_map


def homo_make_proto_tensor_cls():
    make_proto_tensor_cls(["capacity"], [torch.zeros(0, dtype=torch.int64)])


class HomoFusedScatterRouter(ScatterRouter):
    def __init__(
        self,
        dst_num: int,
        route_func,
        route_method: str = "topk",
        residual_dst: int = -1,
        transform: bool = False,
        **kwargs,
    ):
        super().__init__(
            dst_num, route_func, route_method, residual_dst, transform, **kwargs
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
        _in_flow_tag = in_flow_tag_stack.pop()
        _in_flow_load = in_flow_load_stack.pop()

        for key in extra_attr_stack.keys():
            extra_attr_stack[key].pop()

        route_shape = list(in_flow_data.shape[1:])
        route_size = np.prod(route_shape)

        # gates, route_indices: [1 x dst_num]
        active_branch = route_indices.view(-1, 1)
        gates_transform = gates.view(-1, 1)

        all_out_flows_data = F.linear(active_branch, in_flow_data.view(1, -1))
        if self.transform:
            all_out_flows_data = all_out_flows_data * gates_transform
        all_out_flows_data = all_out_flows_data.view(self.dst_num, *route_shape)

        out_flow_load = 1
        active_branch = active_branch.view(-1)

        out_flows: List[ProtoTensor] = []
        for i in range(self.dst_num):
            if active_branch[i] == 0:
                out_flows.append(
                    init_proto_tensor(
                        torch.zeros(
                            (0, *route_shape),
                            dtype=in_flow_data.dtype,
                            device=in_flow_data.device,
                        ),
                        in_flow_tag_stack,
                        in_flow_load_stack,
                        **extra_attr_stack,
                    )
                )
                out_flow_tag = torch.zeros(
                    (0, 1), dtype=torch.int64, device=in_flow_data.device
                )
            else:
                out_flows.append(
                    init_proto_tensor(
                        all_out_flows_data[i],
                        in_flow_tag_stack,
                        in_flow_load_stack,
                        **extra_attr_stack,
                    )
                )
                out_flow_tag = torch.zeros(
                    (1, 1), dtype=torch.int64, device=in_flow_data.device
                )
            out_flows[i].pack(out_flow_tag, out_flow_load, active_branch=active_branch)
        return out_flows

    def remove_needless_pack(self, out_flow):
        return out_flow

    def symbolic_route(self, inputs: torch.Tensor) -> torch.Tensor:
        return symbolic_scatter_route(inputs, 1)[0]


class HeteroFusedGatherRouter(GatherRouter):
    def __init__(self, dst_num: int, reduction: str = "add", sparse=True):
        super().__init__(dst_num=dst_num)
        self.sparse = sparse
        self.reduction = reduction

    def combine(self, in_flows: List[ProtoTensor]) -> Union[ProtoTensor, torch.Tensor]:
        assert len(in_flows) == self.dst_num
        in_flows_data = []
        in_flows_load = []

        flow_tags, flow_loads = [], []
        extra_attr_stack = {}

        for flow in in_flows:
            data, flow_tags, flow_loads, extra_attr_stack = to_torch_tensor(
                flow, copy_stack=True
            )
            in_flows_data.append(data)
            in_flows_load.append(flow_loads.pop())

        in_flows_data = torch.cat(in_flows_data, dim=0)
        in_flows_load = np.max(in_flows_load)

        flow_tags.pop()
        for key in extra_attr_stack.keys():
            extra_attr_stack[key].pop()
        in_flows_tag_stack = flow_tags
        in_flows_load_stack = flow_loads
        in_flows_extra_attr_stack = extra_attr_stack

        out_flow_data = torch.sum(in_flows_data, dim=0, keepdim=True)
        out_flow = init_proto_tensor(
            out_flow_data,
            in_flows_tag_stack,
            in_flows_load_stack,
            **in_flows_extra_attr_stack,
        )
        if out_flow_data.numel() == 0:
            out_flow.pack(
                torch.zeros((0, 1), dtype=torch.int64, device=out_flow_data.device), 1
            )
        else:
            out_flow.pack(
                torch.zeros(1, 1), dtype=torch.int64, device=out_flow_data.device
            )
            
    def symbolic_route(self, inputs: torch.Tensor) -> torch.Tensor:
        return symbolic_gather_route([inputs], 1)
