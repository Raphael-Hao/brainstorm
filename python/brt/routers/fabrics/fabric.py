# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Tuple, Union

import numpy as np
import torch
from brt._C.router import generate_dst_indices
from brt.common import log
from brt.frontend import nn

from ..proto_tensor import (
    ProtoTensor,
    deinit_proto_tensor,
    init_proto_tensor,
    to_torch_tensor,
)

logger = log.get_logger(__file__)


class SwitchFabric(nn.Module):
    def __init__(self, path_num: int) -> None:
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

    def pack_invalid_flow(self, in_flow):
        from ..proto_tensor import ProtoTensor  # we need to keep ProtoTensor updated

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
        from ..proto_tensor import ProtoTensor  # we need to keep ProtoTensor updated

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


class DispatchSF(SwitchFabric):
    def __init__(self, path_num, **kwargs):
        super().__init__(path_num)
        route_logics = kwargs.get("route_logic", ["1d"])
        if isinstance(route_logics, str):
            route_logics = [route_logics]
        assert isinstance(route_logics, list) and all(
            isinstance(x, str) for x in route_logics
        )
        self.route_logics = route_logics
        transforms = kwargs.get("transform", [False])
        if isinstance(transforms, bool):
            transforms = [transforms]
        assert isinstance(transforms, list) and all(
            isinstance(x, bool) for x in transforms
        )
        self.transforms = transforms
        self.indices_gen_opt = kwargs.get("indices_gen_opt", True)

    def forward(
        self,
        in_flow: Union[ProtoTensor, List[ProtoTensor]],
        hot_mask: torch.Tensor,
        score: torch.Tensor,
    ) -> Union[List[ProtoTensor], List[List[ProtoTensor]]]:

        route_indices = self.gen_indices(hot_mask)

        in_flow = self.pack_invalid_flow(in_flow)

        if isinstance(in_flow, ProtoTensor):
            in_flows = [in_flow]
        elif isinstance(in_flow, list):
            in_flows = in_flow
        else:
            raise ValueError("in_flow must be ProtoTensor or list of ProtoTensor")

        all_out_flows = []

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

            for i in range(self.path_num):
                tag_indices = route_indices[i]
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
            if isinstance(in_flow, ProtoTensor):
                return self.remove_needless_pack(out_flows)
            all_out_flows.append(out_flows)

        return self.remove_needless_pack(all_out_flows)

    def gen_indices(self, hot_mask: torch.Tensor) -> List[torch.Tensor]:
        """generate indices according to hot_mask

        Args:
            hot_mask (torch.Tensor): a multi-hot mask for representing the routing destinations of each sample
        """
        if hot_mask.is_cuda and self.indices_gen_opt:
            route_indices = generate_dst_indices(hot_mask)
        else:
            route_indices = []
            hot_mask_t = hot_mask.t().contiguous()
            for i in range(self.path_num):
                route_indices.append(torch.nonzero(hot_mask_t[i].view(-1)))
        return route_indices


class CombineSF(SwitchFabric):
    def __init__(self, path_num, **kwargs):
    
        super().__init__(path_num)
        self.sparse = kwargs.get("sparse", False)
        self.reduction = kwargs.get("reduction", "add")

    def forward(self, in_flows: List[ProtoTensor]) -> ProtoTensor:
        in_flows = self.pack_invalid_flow(in_flows)
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
            ).scatter_(0, route_indices, in_flows_data, reduce=self.reduction)
        out_flow = init_proto_tensor(
            out_flow_data,
            in_flows_tag_stack,
            in_flows_load_stack,
            in_flows_extra_stack_dict,
        )
        out_flow = out_flow.pack(out_flow_tag, out_flow_load)
        return self.remove_needless_pack(out_flow)
