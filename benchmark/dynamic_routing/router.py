# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from brt.routers import ScatterRouter, BaseRouter
from brt.routers.proto_tensor import (
    ProtoTensor,
    to_torch_tensor,
)

__all__ = ["DynamicScatterRouter"]


class DynamicScatterRouter(nn.Module):
    def __init__(
        self, dst_num, residual_dst: int = -1,
    ):
        super().__init__()
        self.sr = ScatterRouter(
            dst_num, None, "threshold", residual_dst, False, threshold=0.0
        )

    def forward(
        self, in_flow: Union[torch.Tensor, ProtoTensor], gates: torch.Tensor
    ) -> List[ProtoTensor]:
        in_flow = self.sr.pack_invalid_flow(in_flow)
        (
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
        ) = to_torch_tensor(in_flow, copy_stack=True)
        assert (
            gates.size(0) == in_flow_data.size(0) and gates.size(1) == self.sr.dst_num
        )
        route_mask = self.gen_mask(gates).to(in_flow.device)
        out_flows = self.sr.dispatch(
            in_flow_data,
            in_flow_tag_stack,
            in_flow_load_stack,
            extra_attr_stack_dict,
            route_mask,
            gates,
        )
        out_flows = self.sr.remove_needless_pack(out_flows)
        return out_flows

    def gen_mask(self, gates: torch.Tensor) -> torch.Tensor:
        # NOTE: inflow.device -> gates.device
        assert self.sr.route_method == "threshold"
        route_hot_mask = (
            (gates > self.sr.threshold).long().to(gates.device)
        )  # [bs x dst_num]
        if self.sr.residual_dst >= 0:
            residual_indices = (
                (route_hot_mask.sum(dim=1, keepdim=True) == 0).long().to(gates.device)
            )  # [bs x 1]
            residual_index = torch.full(
                (residual_indices.shape),
                self.sr.residual_dst,
                dtype=torch.int64,
                device=gates.device,
            )
            route_hot_mask = torch.scatter_add(
                route_hot_mask, 1, residual_index, residual_indices
            )
        return route_hot_mask

