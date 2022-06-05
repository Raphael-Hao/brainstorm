# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from brt.primitive import router

from .flow_tensor import FlowTensor, deinit_flow_tensor, init_flow_tensor


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

    def route(self, *inputs):
        raise NotImplementedError

    def symbolic_route(self, *inputs):
        raise NotImplementedError

    def verify_in_flow(self, in_flow):
        if isinstance(in_flow, FlowTensor):
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
            in_flow = init_flow_tensor(in_flow, [tag], [tag.numel()])

        elif isinstance(in_flow, (List, Tuple)):
            in_flow = type(in_flow)([self.verify_in_flow(f) for f in in_flow])

        else:
            raise ValueError(f"unsupported input type {type(in_flow)}")

        return in_flow
    
    def verify_out_flow(self, out_flow):
        if isinstance(out_flow, FlowTensor):
            if out_flow.size(0) != out_flow.tag.numel():
                # route granularity changed, we will re-tag the inputs
                new_tag = torch.arange(
                    0, out_flow.size(0), dtype=torch.int64, device=out_flow.device
                ).view(-1, 1)
                out_flow.pack(new_tag, load=new_tag.numel())

        elif isinstance(out_flow, torch.Tensor):
            tag = torch.arange(
                0, out_flow.size(0), dtype=torch.int64, device=out_flow.device
            ).view(-1, 1)
            out_flow = init_flow_tensor(out_flow, [tag], [tag.numel()])

        elif isinstance(out_flow, (List, Tuple)):
            out_flow = type(out_flow)([self.verify_out_flow(f) for f in out_flow])

        else:
            raise ValueError(f"unsupported input type {type(out_flow)}")

        return out_flow
