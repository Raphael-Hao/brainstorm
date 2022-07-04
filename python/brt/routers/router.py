# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from brt._C.router import generate_dst_indices
from brt.common import log
from brt.frontend.primitive import router

from .proto_tensor import (
    ProtoTensor,
    deinit_proto_tensor,
    init_proto_tensor,
    to_torch_tensor,
)
from .symbolic import (
    symbolic_gather_route,
    symbolic_joint_gather_route,
    symbolic_joint_scatter_route,
    symbolic_scatter_route,
)

__all__ = [
    "BaseRouter",
    "ScatterRouter",
    "GatherRouter",
]

logger = log.get_logger(__file__)


@router
class BaseRouter(nn.Module):
    def __init__(self, path_num: int):
        """_summary_

        Args:
            path_num (int): number of paths for routing source or destinations
            gran_dim (_type_, optional): routing granularity. should be a int or a list of int.
        """
        super().__init__()
        self.path_num = path_num
        self.start_event = torch.jit.unused(torch.cuda.Event(enable_timing=True))
        self.end_event = torch.jit.unused(torch.cuda.Event(enable_timing=True))

    def start_timer(self):
        self.start_event.record(torch.cuda.current_stream())

    def end_timer(self, timer_name):
        self.end_event.record(torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
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
        path_num: int,
        gate_method: str = "topk",
        route_logic: str = "1d",
        residual_dst: int = -1,
        transform: bool = False,
        **kwargs,
    ):
        """base scatter router

        Args:
            path_num (int): number of paths for routing destinations
            route_method (string): "topk", "threshold"
                topk: select the topk of gate results as the route destinations
                threshold: select the gate results that are larger than threshold as the route destinations
            route_logic (string): "1d", "2d"
                1d: route along the 1st dimension, selecting data from a tensor with shape (batch_size, ...)
                2d: route along the first 2 dimensions, selecting data from a tensor with shape (batch_size, dst_num, ...)
            residual_dst (int): residual destination, if set, the data not selected by gates will all routed to the residual destination.

        """
        super().__init__(path_num=path_num)
        self.gate_method = gate_method.lower()
        self.route_logic = route_logic.lower()
        self.residual_dst = residual_dst
        self.transform = transform
        if self.gate_method == "topk":
            self.k = kwargs.get("k")
            if self.k == None:
                self.k = 1
                logger.warning(
                    "k is not specified for Top-K route method, use default k=1"
                )
        elif self.gate_method == "threshold":
            self.threshold = kwargs.get("threshold")
            if self.threshold == None:
                self.threshold = 0.0
                logger.warning(
                    "threshold is not specified for Threshold route method, use default threshold=0.0"
                )
        else:
            raise ValueError(f"route_method {gate_method} is not supported")

    def route(
        self, in_flow: Union[torch.Tensor, ProtoTensor], score: torch.Tensor
    ) -> Union[List[ProtoTensor], Tuple[List[ProtoTensor], List[ProtoTensor]]]:

        assert score.size(0) == in_flow.size(0)
        assert score.size(1) == self.path_num

        hot_mask = self.gen_hot_mask(score)
        route_indices = self.gen_indices(hot_mask)

        in_flow = self.pack_invalid_flow(in_flow)
        (
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
        ) = to_torch_tensor(in_flow, copy_stack=True)

        out_flows = self.dispatch(
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
            route_indices,
            score,
            self.route_logic,
            self.transform,
        )
        out_flows = self.remove_needless_pack(out_flows)
        return out_flows

    def gen_hot_mask(self, score: torch.Tensor) -> torch.Tensor:
        """generate hotmask with gates
        Args:
            gates (torch.Tensor): output of gating function
        """
        assert score.size(1) == self.path_num
        if self.gate_method == "topk":

            hot_mask = torch.topk(score, self.k, dim=1).indices  # sample x k
            hot_mask = torch.zeros(
                score.size(0), self.path_num, dtype=torch.int64, device=score.device
            ).scatter_(
                1, hot_mask, 1
            )  # sample x dst_num

        elif self.gate_method == "threshold":
            hot_mask = (
                (score > self.threshold).long().to(score.device)
            )  # [bs x dst_num]
            if self.residual_dst >= 0:
                residual_indices = (
                    (hot_mask.sum(dim=1, keepdim=True) == 0).long().to(score.device)
                )  # [bs x 1]
                residual_index = torch.full(
                    (residual_indices.shape),
                    self.residual_dst,
                    dtype=torch.int64,
                    device=score.device,
                )
                hot_mask = torch.scatter_add(
                    hot_mask, 1, residual_index, residual_indices
                )
        else:
            raise ValueError(f"route_method {self.gate_method} is not supported")
        return hot_mask

    def gen_indices(self, hot_mask: torch.Tensor) -> List[torch.Tensor]:
        """generate indices according to hot_mask

        Args:
            hot_mask (torch.Tensor): a multi-hot mask for representing the routing destinations of each sample
        """
        if hot_mask.is_cuda:
            route_indices = generate_dst_indices(hot_mask)
        else:
            route_indices = []
            hot_mask_t = hot_mask.t().contiguous()
            for i in range(self.path_num):
                route_indices.append(torch.nonzero(hot_mask_t[i].view(-1)))
        return route_indices

    def dispatch(
        self,
        in_flow_data: torch.Tensor,
        in_flow_tags: List[torch.Tensor],
        in_flow_loads: List[int],
        extra_attr_stack_dict: Dict[str, List[Any]],
        route_indices: List[torch.Tensor],
        score: torch.Tensor,
        route_logic: str = "1d",
        transform: bool = False,
    ) -> Union[List[ProtoTensor], Tuple[List[ProtoTensor], List[ProtoTensor]]]:  # {dst}
        """
        Dispatch the inputs into the the list of torch.Tensor with indices

        Returns:
            If set routing_gates = True, return a tuple of the routed in_flow and routed gates,
            otherwise only routes in_flow
        """
        in_flow_tag = in_flow_tags.pop()
        in_flow_load = in_flow_loads.pop()

        for attr_stack in extra_attr_stack_dict.values():
            attr_stack.pop()

        if route_logic == "1d":
            route_shape = list(in_flow_data.shape[1:])
            route_size = np.prod(route_shape)
        elif route_logic == "2d":
            in_flow_data = in_flow_data.transpose(0, 1).contiguous()
            route_shape = list(in_flow_data.shape[2:])
            route_size = np.prod(route_shape)

        out_flows = []

        for i in range(self.path_num):
            tag_indices = route_indices[i]
            if tag_indices.numel() > 0:
                out_flow_tag = torch.gather(in_flow_tag, 0, tag_indices)
                data_indices = tag_indices.repeat(1, route_size).view(-1, *route_shape)
                if route_logic == "1d":
                    dispatched_data = in_flow_data
                elif route_logic == "2d":
                    dispatched_data = in_flow_data[i]
                if transform:
                    dispatched_data = dispatched_data * score[:, i].view(
                        (-1,) + (1,) * len(route_shape)
                    )
                out_flow_data = torch.gather(dispatched_data, 0, data_indices)
                out_flow = init_proto_tensor(
                    out_flow_data, in_flow_tags, in_flow_loads, extra_attr_stack_dict
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
                    in_flow_tags,
                    in_flow_loads,
                    extra_attr_stack_dict,
                )
                out_flow.pack(
                    torch.zeros(0, 1, dtype=torch.int64, device=in_flow_data.device),
                    in_flow_load,
                )
            out_flows.append(out_flow)
        return out_flows

    def symbolic_route(
        self, inputs: torch.Tensor, score: torch.Tensor
    ) -> List[torch.Tensor]:
        return symbolic_scatter_route(inputs, self.path_num)


@router
class GatherRouter(BaseRouter):
    def __init__(self, path_num: int, reduction: str = "add", sparse=True):
        """_summary_

        Args:
            path_num (int): number of paths for routing sources
            reduction (str, optional): reduction method. Defaults to "add".
            sparse (bool, optional): whether restore with zero paddings. Defaults to True.
        """
        super().__init__(path_num=path_num)
        self.sparse = sparse
        self.reduction = reduction

    def route(self, in_flows: List[ProtoTensor]) -> ProtoTensor:
        in_flows = self.pack_invalid_flow(in_flows)
        out_flow = self.combine(in_flows, self.reduction)
        out_flow = self.remove_needless_pack(out_flow)
        return out_flow

    def combine(self, in_flows: List[ProtoTensor], reduction: str) -> ProtoTensor:
        """
        Combine the outputs of the routers into the final outputs
        """
        assert len(in_flows) == self.path_num
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
            ).scatter_(0, route_indices, in_flows_data, reduce=reduction)
        out_flow = init_proto_tensor(
            out_flow_data,
            in_flows_tag_stack,
            in_flows_load_stack,
            in_flows_extra_stack_dict,
        )
        out_flow = out_flow.pack(out_flow_tag, out_flow_load)
        return out_flow

    def symbolic_route(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return symbolic_gather_route(inputs, self.path_num)


@router
class JointScatterRouter(ScatterRouter):
    def __init__(
        self,
        path_num: int,
        gate_method: str = "topk",
        route_logics: List[str] = ["1d"],
        transforms: List[bool] = [False],
        residual_dst: int = -1,
        **kwargs,
    ):
        """base scatter router

        Args:
            path_num (int): routing number
            gate_method = "topk", "threshold"
                topk: select the topk of gate results as the route destinations
                threshold: select the gate results that are larger than threshold as the route destinations
            dispatcher_cls (type): dispatcher class
        """
        super().__init__(
            path_num,
            gate_metho=gate_method,
            residual_dst=residual_dst,
            transform=False,
            **kwargs,
        )
        self.route_logics = [logic.lower() for logic in route_logics]
        self.transforms = transforms
        assert len(self.route_logics) == len(self.transforms)
        self.in_flow_num = len(self.route_logic)

    def route(
        self,
        in_flows: List[ProtoTensor],
        score: torch.Tensor,
    ) -> List[List[ProtoTensor]]:

        assert isinstance(
            in_flows, List
        ), "in_flows must be a list for joint scatter router"
        assert (
            len(in_flows) == self.in_flow_num
        ), "in_flows must be a list of length in_flow_num"

        hot_mask = self.gen_hot_mask(score)
        route_indices = self.gen_indices(hot_mask)

        in_flows = self.pack_invalid_flow(in_flows)

        out_flows = []
        for i, in_flow in enumerate(in_flows):
            (
                in_flow_data,
                in_flow_tag_stack,
                in_flow_load_stack,
                extra_attr_stack_dict,
            ) = to_torch_tensor(in_flow, copy_stack=True)

            out_flows.append(
                self.dispatch(
                    in_flow_data,
                    in_flow_tag_stack,
                    in_flow_load_stack,
                    extra_attr_stack_dict,
                    route_indices,
                    score,
                    self.route_logic[i],
                    self.transforms[i],
                )
            )

        out_flows = self.remove_needless_pack(out_flows)
        return out_flows

    def symbolic_route(
        self, inputs: List[torch.Tensor], score: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        return symbolic_joint_scatter_route(inputs, self.path_num)


@router
class JointGatherRouter(GatherRouter):
    def __init__(self, dst_num: int, reductions: List[str] = ["add"], sparse=True):
        self.reductions = [reduction.lower() for reduction in reductions]
        self.in_flow_num = len(self.reductions)
        super().__init__(path_num=dst_num, sparse=sparse)

    def route(self, in_flows: List[List[ProtoTensor]]) -> ProtoTensor:
        assert len(in_flows) == self.in_flow_num
        in_flows = self.pack_invalid_flow(in_flows)
        out_flows = []
        for i, in_flow in enumerate(in_flows):
            out_flows.append(self.combine(in_flow, self.reductions[i]))
        out_flow = self.remove_needless_pack(out_flow)
        return out_flow

    def symbolic_route(self, inputs: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        return symbolic_joint_gather_route(inputs, self.path_num)
