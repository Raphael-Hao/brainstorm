# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch
from brt.runtime import log
from brt.router.utils import pad_to_max
from brt.router.fabric.base import FabricBase, register_fabric
from brt.runtime.proto_tensor import ProtoTensor, init_proto_tensor, to_torch_tensor

logger = log.get_logger(__file__)


@register_fabric("dispatch")
class DispatchFabric(FabricBase):
    def __init__(
        self,
        flow_num: int,
        throttling=False,
        route_logic: Union[str, List[str]] = "1d",
        transform: Union[bool, List[bool]] = False,
        **kwargs,
    ):
        """dispatch fabric

        Args:
            kwargs (dict):
                route_logic (str): 1d or 2d, default is 1d, can be list of 1d or 2d
                transform (bool): whether to transform input with the score, default is False, can be list of bool
        """
        index_format = kwargs.pop("index_format", "src_index")
        super().__init__(flow_num=flow_num, index_format=index_format, **kwargs)
        self.throttling = throttling
        route_logics = route_logic
        if isinstance(route_logics, str):
            assert route_logics in ["1d", "2d"]
            route_logics = [route_logics]
        assert isinstance(route_logics, list) and all(
            isinstance(x, str) and x in ["1d", "2d"] for x in route_logics
        )
        transforms = transform
        if isinstance(transforms, bool):
            transforms = [transforms]
        assert isinstance(transforms, list) and all(
            isinstance(x, bool) for x in transforms
        )

        assert len(route_logics) == len(transforms)
        assert self.flow_num == len(route_logics)
        self.route_logics = route_logics
        self.transforms = transforms

    def forward(
        self,
        in_flow: Union[ProtoTensor, List[ProtoTensor]],
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        capacities: torch.Tensor = None,
        score: torch.Tensor = None,
    ) -> Union[List[ProtoTensor], List[List[ProtoTensor]]]:
        assert self.index_format == "src_index", "DispatchFabric only support src_index"
        in_flow = self.pack_invalid_flow(in_flow)

        if self.flow_num == 1:
            in_flows = [in_flow]
        else:
            in_flows = in_flow

        if self.throttling:
            real_loads = torch.minimum(loads, capacities)
        else:
            real_loads = loads
        all_out_flows = self.dispatch(in_flows, route_indices, real_loads, score)
        if self.flow_num == 1:
            return self.remove_needless_pack(all_out_flows[0])
        return self.remove_needless_pack(all_out_flows)

    def dispatch(
        self,
        in_flows: List[ProtoTensor],
        route_indices: torch.Tensor,
        real_loads: torch.Tensor,
        score: torch.Tensor,
    ) -> List[List[ProtoTensor]]:
        all_out_flows = []
        path_num = route_indices.size(1)
        # real_loads = real_loads.cpu()
        route_indices_64 = route_indices.to(torch.int64)
        for flow_idx in range(self.flow_num):
            flow = in_flows[flow_idx]
            (
                flow_data,
                flow_tag_stack,
                flow_load_stack,
                extra_attr_stack_dict,
            ) = to_torch_tensor(flow, return_stack=True)

            flow_tag = flow_tag_stack[-1]
            flow_load = flow_load_stack[-1]

            flow_tag_stack = flow_tag_stack[:-1]
            flow_load_stack = flow_load_stack[:-1]
            flow_extra_attr_stack_dict = {}
            for attr_dict, attr_stack in extra_attr_stack_dict.items():
                flow_extra_attr_stack_dict[attr_dict] = attr_stack[:-1]

            if self.route_logics[flow_idx] == "1d":
                route_shape = list(flow_data.shape[1:])
                # route_size = np.prod(route_shape)
            elif self.route_logics[flow_idx] == "2d":
                # flow_data = flow_data.transpose(0, 1).contiguous()
                route_shape = list(flow_data.shape[1:])
                route_shape[0] = 1
                # route_size = np.prod(route_shape)
            else:
                raise ValueError("route_logic must be 1d or 2d")
            out_flows = []
            for i in range(path_num):
                tag_indices = route_indices_64[: real_loads[i], i : i + 1]

                if tag_indices.numel() > 0:
                    out_flow_tag = torch.gather(flow_tag, 0, tag_indices)
                    if self.route_logics[flow_idx] == "1d":
                        dispatched_data = flow_data
                    elif self.route_logics[flow_idx] == "2d":
                        dispatched_data = flow_data[:, i : i + 1].contiguous()
                    else:
                        raise ValueError("route_logic must be 1d or 2d")
                    if self.transforms[flow_idx]:
                        dispatched_data = dispatched_data * score[:, i].view(
                            (-1,) + (1,) * len(route_shape)
                        )
                    out_flow_data = torch.index_select(
                        dispatched_data, 0, tag_indices.view(-1)
                    )
                    out_flow = init_proto_tensor(
                        out_flow_data,
                        flow_tag_stack,
                        flow_load_stack,
                        flow_extra_attr_stack_dict,
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
                        flow_extra_attr_stack_dict,
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
    def __init__(
        self,
        flow_num: int,
        reduction="add",
        sparse=False,
        granularity_padding=False,
        **kwargs,
    ):
        super().__init__(flow_num=flow_num, index_format="src_index", **kwargs)

        self.sparse = sparse
        self.reduction = reduction
        self.granularity_padding = granularity_padding

    def forward(
        self, in_flows: Union[List[ProtoTensor], List[List[ProtoTensor]]]
    ) -> Union[ProtoTensor, List[ProtoTensor]]:
        in_flows = self.pack_invalid_flow(in_flows)

        if self.flow_num == 1:
            in_flows = [in_flows]

        out_flows = self.combine(in_flows)

        if self.flow_num == 1:
            return self.remove_needless_pack(out_flows[0])

        return self.remove_needless_pack(out_flows)

    def combine(self, in_flows: List[List[ProtoTensor]]) -> List[ProtoTensor]:

        out_flows = []

        for flow_idx in range(self.flow_num):
            in_flow = in_flows[flow_idx]

            in_flow = self.granularity_pad(in_flow)

            in_flows_data = []
            in_flows_tag = []
            in_flows_load = []

            for flow in in_flow:
                data, flow_tags, flow_loads, extra_attr_stack_dict = to_torch_tensor(
                    flow, return_stack=True
                )
                in_flows_data.append(data)
                in_flows_tag.append(flow_tags[-1])
                in_flows_load.append(flow_loads[-1])

            in_flows_data = torch.cat(in_flows_data, dim=0)
            in_flows_tag = torch.cat(in_flows_tag, dim=0)
            in_flows_load = np.max(in_flows_load)
            in_flows_tag_stack = flow_tags[:-1]
            in_flows_load_stack = flow_loads[:-1]
            in_flows_extra_stack_dict = {}
            for attr_dict, attr_stack in extra_attr_stack_dict.items():
                in_flows_extra_stack_dict[attr_dict] = attr_stack[:-1]

            route_shape = list(in_flows_data.shape[1:])
            route_size = np.prod(route_shape)

            if in_flows_data.numel() == 0:
                out_flow_data = torch.zeros(
                    0,
                    *route_shape,
                    dtype=in_flows_data.dtype,
                    device=in_flows_data.device,
                )
                out_flow_tag = in_flows_tag
                out_flow_load = in_flows_load
            else:
                if self.sparse:
                    out_flow_tag, inverse = torch.unique(
                        in_flows_tag, return_inverse=True
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
            out_flow = out_flow.pack(out_flow_tag, in_flows_load)
            out_flows.append(out_flow)

        return out_flows

    def granularity_pad(self, flows):
        if self.granularity_padding:
            flows = pad_to_max(flows, 0)
        return flows
