# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from brt.common import log
from brt.primitive import router

from .proto_tensor import (
    ProtoTensor,
    deinit_proto_tensor,
    init_proto_tensor,
    to_torch_tensor,
)
from .symbolic import symbolic_gather_route, symbolic_scatter_route

__all__ = [
    "BaseRouter",
    "ScatterRouter",
    "GatherRouter",
]

logger = log.get_logger(__file__)


@router
class BaseRouter(nn.Module):
    def __init__(self, dst_num: int):
        """_summary_

        Args:
            dst_num (int): number of src or dst for routing
            gran_dim (_type_, optional): routing granularity. should be a int or a list of int.
        """
        super().__init__()
        self.dst_num = dst_num
        self.start_event = torch.jit.unused(torch.cuda.Event(enable_timing=True))
        self.end_event = torch.jit.unused(torch.cuda.Event(enable_timing=True))

    def start_timer(self):
        self.start_event.record(torch.cuda.current_stream())

    def end_timer(self, timer_name):
        self.end_event.record(torch.cuda.current_stream())
        self.stream.synchronize()
        print(
            "{} elapsed time: {:.3f}".format(
                timer_name, self.start_event.elapsed_time(self.end_event)
            )
        )

    def route(self, *inputs):
        raise NotImplementedError

    def symbolic_route(self, *inputs):
        raise NotImplementedError

    def pack_invalid_flow(self, in_flow):
        from .proto_tensor import ProtoTensor  # we need to keep ProtoTensor updated

        if isinstance(in_flow, ProtoTensor):
            if in_flow.size(0) != in_flow.tag.numel():
                # route granularity changed, we will re-tag the inputs
                new_tag = torch.arange(
                    0, in_flow.size(0), dtype=torch.int64, device=in_flow.device
                ).view(-1, 1)
                in_flow.pack(new_tag, load=new_tag.numel())

        elif isinstance(in_flow, torch.Tensor):
            tag = torch.arange(
                0, in_flow.size(0), dtype=torch.int64, device=in_flow.device
            ).view(-1, 1)
            in_flow = init_proto_tensor(in_flow, [tag], [tag.numel()])

        elif isinstance(in_flow, (List, Tuple)):
            in_flow = type(in_flow)([self.pack_invalid_flow(f) for f in in_flow])

        return in_flow

    def remove_needless_pack(self, out_flow):
        from .proto_tensor import ProtoTensor  # we need to keep ProtoTensor updated

        if isinstance(out_flow, ProtoTensor):
            if out_flow.proto_empty():
                out_flow, _, _, _ = deinit_proto_tensor(out_flow)
            elif out_flow.tag.numel() == out_flow.load:
                out_flow, _, _, _ = out_flow.unpack()
                if out_flow.proto_empty():
                    out_flow, _, _, _ = deinit_proto_tensor(out_flow)

        elif isinstance(out_flow, (List, Tuple)):
            out_flow = type(out_flow)([self.remove_needless_pack(f) for f in out_flow])

        return out_flow


@router
class ScatterRouter(BaseRouter):
    def __init__(
        self,
        dst_num,
        route_func,
        route_method: str = "topk",
        residual_dst: int = -1,
        transform=True,
        **kwargs,
    ):
        """base scatter router

        Args:
            dst_num (int): routing number
            route_method = "topk", "threshold"
                topk: select the topk of gate results as the route destinations
                threshold: select the gate results that are larger than threshold as the route destinations
            dispatcher_cls (type): dispatcher class
        """
        super().__init__(dst_num=dst_num)
        self.route_func = route_func
        self.route_method = route_method.lower()
        self.residual_dst = residual_dst
        self.transform = transform
        if self.route_method == "topk":
            self.k = kwargs.get("k")
            if self.k == None:
                self.k = 1
                logger.warning(
                    "k is not specified for Top-K route method, use default k=1"
                )
        elif self.route_method == "threshold":
            self.threshold = kwargs.get("threshold")
            if self.threshold == None:
                self.threshold = 0.0
                logger.warning(
                    "threshold is not specified for Threshold route method, use default threshold=0.0"
                )
        else:
            raise ValueError(f"route_method {route_method} is not supported")

    def route(self, in_flow: Union[torch.Tensor, ProtoTensor]) -> List[ProtoTensor]:
        in_flow = self.pack_invalid_flow(in_flow)
        (
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
        ) = to_torch_tensor(in_flow, copy_stack=True)
        route_mask, gates = self.gen_gates_and_mask(in_flow_data)
        out_flows = self.dispatch(
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
            route_mask,
            gates,
        )
        out_flows = self.remove_needless_pack(out_flows)
        return out_flows

    def gen_gates_and_mask(
        self, in_flow_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """generate gates with indices

        Args:
            inputs (torch.Tensor): input tensor
        """
        gates = self.route_func(in_flow_data)  # sample x dst_num
        assert gates.size(0) == in_flow_data.size(0) and gates.size(1) == self.dst_num
        if self.route_method == "topk":

            route_hot_mask = torch.topk(gates, self.k, dim=1).indices  # sample x k
            route_hot_mask = torch.zeros(
                gates.size(0),
                self.dst_num,
                dtype=torch.int64,
                device=in_flow_data.device,
            ).scatter_(
                1, route_hot_mask, 1
            )  # sample x dst_num

        elif self.route_method == "threshold":
            route_hot_mask = (
                (gates > self.threshold).long().to(in_flow_data.device)
            )  # [bs x dst_num]
            if self.residual_dst >= 0:
                residual_indices = (
                    (route_hot_mask.sum(dim=1, keepdim=True) == 0)
                    .long()
                    .to(in_flow_data.device)
                )  # [bs x 1]
                residual_index = torch.full(
                    (residual_indices.shape),
                    self.residual_dst,
                    dtype=torch.int64,
                    device=in_flow_data.device,
                )
                route_hot_mask = torch.scatter_add(
                    route_hot_mask, 1, residual_index, residual_indices
                )
        return route_hot_mask, gates

    def dispatch(
        self,
        in_flow_data: torch.Tensor,
        in_flow_tags: List[torch.Tensor],
        in_flow_loads: List[int],
        extra_attr_stack_dict: Dict[str, List[Any]],
        route_mask: torch.Tensor,
        gates: torch.Tensor,
    ) -> List[ProtoTensor]:  # {dst}
        """
        Dispatch the inputs into the the list of torch.Tensor with indices
        """

        in_flow_tag = in_flow_tags.pop()
        in_flow_load = in_flow_loads.pop()

        route_shape = list(in_flow_data.shape[1:])
        route_size = np.prod(route_shape)
        gates_T = gates.t()  # [dst_num x sample]
        route_mask_T = route_mask.t()  # [dst_num x sample]

        out_flows = []
        for i in range(self.dst_num):
            tag_indices = torch.nonzero(
                route_mask_T[i].view(-1)
            )  # TODO torch.nonzero will cause hostside synchronization, we need a async one
            if tag_indices.numel() > 0:
                out_flow_tag = torch.gather(in_flow_tag, 0, tag_indices)
                data_indices = tag_indices.repeat(1, route_size).view(-1, *route_shape)
                if self.transform:
                    gate = gates_T[i].reshape(
                        (in_flow_data.size(0),) + (1,) * len(route_shape)
                    )
                    out_flow_data = torch.gather(in_flow_data * gate, 0, data_indices)
                else:
                    out_flow_data = torch.gather(in_flow_data, 0, data_indices)
                out_flow = init_proto_tensor(out_flow_data, in_flow_tags, in_flow_loads)
                out_flow.pack(out_flow_tag, in_flow_load)
            else:
                out_flow = init_proto_tensor(
                    torch.zeros(
                        0,
                        *route_shape,
                        dtype=in_flow_data.dtype,
                        device=in_flow_data.device,
                    ),
                    in_flow_tags,
                    in_flow_loads,
                )
                out_flow.pack(
                    torch.zeros(0, 1, dtype=torch.int64, device=in_flow_data.device),
                    in_flow_load,
                )
            out_flows.append(out_flow)
        return out_flows

    def symbolic_route(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return symbolic_scatter_route(inputs, self.dst_num)


@router
class GatherRouter(BaseRouter):
    def __init__(
        self, dst_num: int, reduction: str = "add", sparse=True, transform=False
    ):
        super().__init__(dst_num=dst_num)
        self.transform = transform
        self.sparse = sparse
        self.reduction = reduction

    def route(self, in_flows: List[ProtoTensor]) -> ProtoTensor:
        in_flows = self.pack_invalid_flow(in_flows)
        out_flow = self.combine(in_flows)
        out_flow = self.remove_needless_pack(out_flow)
        return out_flow

    def combine(self, in_flows: List[ProtoTensor]) -> Union[ProtoTensor, torch.Tensor]:
        """
        Combine the outputs of the routers into the final outputs
        """
        assert len(in_flows) == self.dst_num
        in_flows_data = []
        in_flows_tag = []
        in_flows_load = []
        flow_tags, flow_loads = [], []

        for flow in in_flows:
            data, flow_tags, flow_loads, _ = to_torch_tensor(flow, copy_stack=True)
            in_flows_data.append(data)
            in_flows_tag.append(flow_tags.pop())
            in_flows_load.append(flow_loads.pop())
        in_flows_data = torch.cat(in_flows_data, dim=0)
        in_flows_tag = torch.cat(in_flows_tag, dim=0)
        in_flows_load = np.max(in_flows_load)
        in_flows_tags = flow_tags
        in_flows_loads = flow_loads

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
        out_flow_tags = in_flows_tags + [out_flow_tag]
        out_flow_loads = in_flows_loads + [in_flows_load]
        return init_proto_tensor(out_flow_data, out_flow_tags, out_flow_loads)

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        return symbolic_gather_route(inputs, self.dst_num)
