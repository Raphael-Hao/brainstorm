# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union, List, Dict, Any

import torch
from brt.router.fabric.generic import DispatchFabric, CombineFabric
from brt.router.fabric.base import register_fabric
from brt.runtime.proto_tensor import ProtoTensor, init_proto_tensor


@register_fabric("single_ptu_dispatch")
class SinglePTUDispatchFabric(DispatchFabric):
    def __init__(
        self,
        flow_num: int,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        **kwargs
    ):
        super().__init__(
            flow_num=flow_num, route_logic=route_logic, transform=transform
        )
        self.check_compatibility(kwargs)

    def forward(
        self,
        in_flow: Union[torch.Tensor, List[torch.Tensor]],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        capacities: torch.Tensor = None,
        score: torch.Tensor = None,
    ) -> Union[List[torch.Tensor], List[List[torch.Tensor]]]:
        if self.flow_num == 1:
            in_flows = [in_flow]
        else:
            in_flows = in_flow
        all_out_flows = self.dispatch(in_flows, route_indices, loads, score)
        if self.flow_num == 1:
            return all_out_flows[0]
        return all_out_flows

    def dispatch(
        self,
        in_flows: List[torch.Tensor],
        route_indices: torch.Tensor,
        real_loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[List[torch.Tensor]]:
        all_out_flows = []
        path_num = route_indices.size(1)
        for flow_idx in range(self.flow_num):
            flow = in_flows[flow_idx]
            if_proto_tensor = False
            if isinstance(flow, ProtoTensor):
                (
                    flow_data,
                    tag_stack,
                    load_stack,
                    extra_attr_stack_dict,
                ) = flow.copy_stacks()
                if_proto_tensor = True
            else:
                flow_data = flow
            if self.route_logics[flow_idx] == "1d":
                route_shape = list(flow_data.shape[1:])
            elif self.route_logics[flow_idx] == "2d":
                route_shape = list(flow_data.shape[1:])
                route_shape[0] = 1
            else:
                raise ValueError("route_logic must be 1d or 2d")
            out_flows = []
            for path_id, load in enumerate(real_loads):
                if load == 0:
                    if if_proto_tensor:
                        out_flow = init_proto_tensor(
                            torch.zeros(
                                0,
                                *route_shape,
                                dtype=flow_data.dtype,
                                device=flow_data.device,
                            ),
                            tag_stack,
                            load_stack,
                            extra_attr_stack_dict,
                        )
                    else:
                        out_flow = torch.zeros(
                            0,
                            *route_shape,
                            dtype=flow_data.dtype,
                            device=flow_data.device,
                        )
                else:
                    if self.route_logics[flow_idx] == "1d":
                        dispatched_flow = flow
                    elif self.route_logics[flow_idx] == "2d":
                        dispatched_flow = flow[:, path_id:path_id + 1].contiguous()
                    else:
                        raise ValueError("route_logic must be 1d or 2d")

                    if self.transforms[flow_idx]:
                        out_flow = dispatched_flow * score[:, path_id].view(
                            (-1,) + (1,) * len(route_shape)
                        )
                    else:
                        out_flow = dispatched_flow
                out_flows.append(out_flow)
            all_out_flows.append(out_flows)
        return all_out_flows

    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        throttling = kwargs.get("throttling", False)
        if throttling:
            self.assert_compatibility("throttling", False, True)


@register_fabric("single_ptu_combine")
class SinglePTUCombineFabric(CombineFabric):
    def __init__(
        self, flow_num: int, reduction="add", granularity_padding=False, **kwargs
    ):
        self.check_compatibility(kwargs)
        super().__init__(
            flow_num=flow_num,
            reduction=reduction,
            sparse=True,
            granularity_padding=granularity_padding,
        )

    def forward(
        self, in_flows: Union[List[ProtoTensor], List[List[ProtoTensor]]]
    ) -> Union[ProtoTensor, List[ProtoTensor]]:
        if self.flow_num == 1:
            in_flows = [in_flows]

        out_flows = self.combine(in_flows)

        if self.flow_num == 1:
            return out_flows[0]

        return out_flows

    def combine(self, in_flows: List[List[ProtoTensor]]) -> List[ProtoTensor]:
        out_flows = []

        for flow_idx in range(self.flow_num):
            out_flow = torch.cat(in_flows[flow_idx], dim=0).sum(dim=0, keepdim=True)
            out_flows.append(out_flow)

        return out_flows

    def check_compatibility(self, kwargs) -> None:
        sparse = kwargs.get("sparse", True)
        if not sparse:
            self.assert_compatibility("sparse", True, False)
