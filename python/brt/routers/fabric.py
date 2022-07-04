# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any

import numpy as np
import torch
from brt.common import log
from torch.autograd import Function

from .proto_tensor import (
    ProtoTensor,
    deinit_proto_tensor,
    init_proto_tensor,
    to_torch_tensor,
)

logger = log.get_logger(__file__)


def factory_fabric():
    pass


class Fabric(Function):
    pass


class Dispatch(Fabric):
    @staticmethod
    def forward(ctx: Any, in_flow, score, route_indices, router) -> Any:
        (
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
        ) = to_torch_tensor(in_flow, copy_stack=True)

        in_flow_tag = in_flow_tag_stack.pop()
        in_flow_load = in_flow_load_stack.pop()

        for attr_stack in extra_attr_stack_dict.values():
            attr_stack.pop()

        if router.route_logic == "1d":
            route_shape = list(in_flow_data.shape[1:])
            route_size = np.prod(route_shape)
        elif router.route_logic == "2d":
            in_flow_data = in_flow_data.transpose(0, 1).contiguous()
            route_shape = list(in_flow_data.shape[2:])
            route_size = np.prod(route_shape)

        out_flows = []

        for i in range(router.path_num):
            tag_indices = route_indices[i]
            if tag_indices.numel() > 0:
                out_flow_tag = torch.gather(in_flow_tag, 0, tag_indices)
                data_indices = tag_indices.repeat(1, route_size).view(-1, *route_shape)
                if router.route_logic == "1d":
                    dispatched_data = in_flow_data
                elif router.route_logic == "2d":
                    dispatched_data = in_flow_data[i]
                if router.transform:
                    dispatched_data = dispatched_data * score[:, i].view(
                        (-1,) + (1,) * len(route_shape)
                    )
                out_flow_data = torch.gather(dispatched_data, 0, data_indices)
                out_flow = init_proto_tensor(
                    out_flow_data,
                    in_flow_tag_stack,
                    in_flow_load_stack,
                    extra_attr_stack_dict,
                )
                out_flow.pack(out_flow_tag, in_flow_load)
            else:
                out_flow = init_proto_tensor(
                    torch.zeros(
                        0,
                        *route_shape,
                        dtype=in_flow_data.dtype,
                        device=in_flow_data.device,
                    ),
                    in_flow_tag_stack,
                    in_flow_load_stack,
                    extra_attr_stack_dict,
                )
                out_flow.pack(
                    torch.zeros(0, 1, dtype=torch.int64, device=in_flow_data.device),
                    in_flow_load,
                )
            out_flows.append(out_flow)
        return out_flows

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError()


class Combine(Fabric):
    @staticmethod
    def forward(ctx: Any, in_flows, router) -> Any:
        assert len(in_flows) == router.path_num
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
            if router.sparse:
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
            ).scatter_(0, route_indices, in_flows_data, reduce=router.reduction)
        out_flow = init_proto_tensor(
            out_flow_data,
            in_flows_tag_stack,
            in_flows_load_stack,
            in_flows_extra_stack_dict,
        )
        out_flow = out_flow.pack(out_flow_tag, out_flow_load)
        return out_flow


class HomoFusedDispatch(Fabric):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        return super().forward(ctx, *args, **kwargs)


class HomoFusedCombine(Fabric):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        return super().forward(ctx, *args, **kwargs)
