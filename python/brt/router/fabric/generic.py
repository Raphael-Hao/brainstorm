# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch
from brt.common import log
from brt.router.fabric.base import FabricBase, register_fabric
from brt.router.proto_tensor import ProtoTensor, init_proto_tensor, to_torch_tensor

logger = log.get_logger(__file__)


@register_fabric("dispatch")
class DispatchFabric(FabricBase):
    def __init__(self, **kwargs):
        """dispatch fabric

        Args:
            kwargs (dict):
                route_logic (str): 1d or 2d, default is 1d, can be list of 1d or 2d
                transform (bool): whether to transform input with the score, default is False, can be list of bool
        """
        super().__init__(indices_format="src_index")
        self.throttling = kwargs.get("throttling", False)

        route_logics = kwargs.get("route_logic", ["1d"])
        if isinstance(route_logics, str):
            assert route_logics in ["1d", "2d"]
            route_logics = [route_logics]
        assert isinstance(route_logics, list) and all(
            isinstance(x, str) and x in ["1d", "2d"] for x in route_logics
        )
        transforms = kwargs.get("transform", [False])
        if isinstance(transforms, bool):
            transforms = [transforms]
        assert isinstance(transforms, list) and all(
            isinstance(x, bool) for x in transforms
        )
        assert len(route_logics) == len(transforms)

        self.route_logics = route_logics
        self.transforms = transforms

    def forward(
        self,
        in_flow: Union[ProtoTensor, List[ProtoTensor]],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        capacities: torch.Tensor,
        score: torch.Tensor,
    ) -> Union[List[ProtoTensor], List[List[ProtoTensor]]]:

        in_flow = self.pack_invalid_flow(in_flow)

        if isinstance(in_flow, ProtoTensor):
            in_flows = [in_flow]
        elif isinstance(in_flow, list):
            in_flows = in_flow
        else:
            raise ValueError("in_flow must be ProtoTensor or list of ProtoTensor")

        if self.throttling:
            real_loads = torch.minimum(loads, capacities)
        else:
            real_loads = loads
        all_out_flows = self.dispatch(in_flows, route_indices, real_loads, score)
        if isinstance(in_flow, ProtoTensor):
            return self.remove_needless_pack(all_out_flows[0])
        return self.remove_needless_pack(all_out_flows)

    def dispatch(
        self,
        in_flows: List[ProtoTensor],
        route_indices: torch.Tensor,
        real_loads: torch.Tensor,
        score: torch.Tensor,
    ):
        all_out_flows = []
        path_num = route_indices.size(1)
        
        for flow_idx, flow in enumerate(in_flows):
            (
                flow_data,
                flow_tag_stack,
                flow_load_stack,
                extra_attr_stack_dict,
            ) = to_torch_tensor(flow, copy_stack=True)

            flow_tag = flow_tag_stack.pop()
            flow_load = flow_load_stack.pop()

            for attr_stack in extra_attr_stack_dict.values():
                attr_stack.pop()

            if self.route_logics[flow_idx] == "1d":
                route_shape = list(flow_data.shape[1:])
                route_size = np.prod(route_shape)
            elif self.route_logics[flow_idx] == "2d":
                flow_data = flow_data.transpose(0, 1).contiguous()
                route_shape = list(flow_data.shape[2:])
                route_size = np.prod(route_shape)
            else:
                raise ValueError("route_logic must be 1d or 2d")

            out_flows = []
            for i in range(path_num):
                tag_indices = route_indices[: real_loads[i], i]
                if tag_indices.numel() > 0:
                    out_flow_tag = torch.gather(flow_tag, 0, tag_indices)
                    data_indices = tag_indices.repeat(1, route_size).view(
                        -1, *route_shape
                    )
                    if self.route_logics[flow_idx] == "1d":
                        dispatched_data = flow_data
                    elif self.route_logics[flow_idx] == "2d":
                        dispatched_data = flow_data[i]
                    if self.transforms[flow_idx]:
                        dispatched_data = dispatched_data * score[:, i].view(
                            (-1,) + (1,) * len(route_shape)
                        )
                    out_flow_data = torch.gather(dispatched_data, 0, data_indices)
                    out_flow = init_proto_tensor(
                        out_flow_data,
                        flow_tag_stack,
                        flow_load_stack,
                        extra_attr_stack_dict,
                    )
                    out_flow.pack(out_flow_tag, flow_load)
                else:
                    out_flow = init_proto_tensor(
                        torch.zeros(
                            0,
                            *route_shape,
                            dtype=flow_data.dtype,
                            device=flow_data.device,
                        ),
                        flow_tag_stack,
                        flow_load_stack,
                        extra_attr_stack_dict,
                    )
                    out_flow.pack(
                        torch.zeros(0, 1, dtype=torch.int64, device=flow_data.device),
                        flow_load,
                    )
                out_flows.append(out_flow)
            all_out_flows.append(out_flows)
        return all_out_flows


@register_fabric("combine")
class CombineFabric(FabricBase):
    def __init__(self, **kwargs):

        super().__init__(indices_format="src_index")
        self.sparse = kwargs.get("sparse", False)
        self.reduction = kwargs.get("reduction", "add")

    def forward(self, in_flows: List[ProtoTensor]) -> ProtoTensor:
        in_flows = self.pack_invalid_flow(in_flows)
        in_flows_data = []
        in_flows_tag = []
        in_flows_load = []

        for flow in in_flows:
            data, flow_tags, flow_loads, extra_attr_stack_dict = to_torch_tensor(
                flow, copy_stack=True
            )
            in_flows_data.append(data)
            in_flows_tag.append(flow_tags.pop())
            in_flows_load.append(flow_loads.pop())

        for attr_stack in extra_attr_stack_dict.values():
            attr_stack.pop()

        in_flows_data = torch.cat(in_flows_data, dim=0)
        in_flows_tag = torch.cat(in_flows_tag, dim=0)
        in_flows_load = np.max(in_flows_load)
        in_flows_tag_stack = flow_tags
        in_flows_load_stack = flow_loads
        in_flows_extra_stack_dict = extra_attr_stack_dict

        route_shape = list(in_flows_data.shape[1:])
        route_size = np.prod(route_shape)

        if in_flows_data.numel() == 0:
            out_flow_data = torch.zeros(
                0, *route_shape, dtype=in_flows_data.dtype, device=in_flows_data.device
            )
            out_flow_tag = in_flows_tag
        else:
            if self.sparse:
                out_flow_tag, inverse = torch.unique(
                    in_flows_tag, sorted=True, return_inverse=True
                )
                route_indices = inverse.repeat(1, route_size).view(-1, *route_shape)
                out_flow_tag = out_flow_tag.view(-1, 1)
                out_flow_load = out_flow_tag.numel()
            else:
                route_indices = in_flows_tag.repeat(1, route_size).view(
                    -1, *route_shape
                )
                out_flow_tag = torch.arange(
                    0, in_flows_load, dtype=torch.int64, device=in_flows_data.device
                ).view(-1, 1)
                out_flow_load = in_flows_load
            out_flow_data = torch.zeros(
                out_flow_load,
                *route_shape,
                dtype=in_flows_data.dtype,
                device=in_flows_data.device,
            ).scatter_(0, route_indices, in_flows_data, reduce=self.reduction)
        out_flow = init_proto_tensor(
            out_flow_data,
            in_flows_tag_stack,
            in_flows_load_stack,
            in_flows_extra_stack_dict,
        )
        out_flow = out_flow.pack(out_flow_tag, out_flow_load)

        return self.remove_needless_pack(out_flow)
