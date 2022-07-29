# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, List, Tuple, Any

import torch
import torch.nn as nn
from brt.runtime import log
from brt.router.utils import make_kwargs
from brt.runtime.proto_tensor import deinit_proto_tensor, init_proto_tensor
from brt.runtime.registry import Registry

logger = log.get_logger(__file__)


class FabricBase(nn.Module):
    def __init__(self, index_format: str) -> None:
        super().__init__()
        self.index_format = index_format
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        assert self.index_format in [
            "dst_index",
            "src_index",
        ], f"index_format should be dst_index or src_index, but got {index_format}"

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

    def forward(
        self,
        in_flow,
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        capacities: torch.Tensor,
        score: torch.Tensor,
    ):
        raise NotImplementedError("FabricBase is an abstract class for Fabric")

    def pack_invalid_flow(self, in_flow):
        from brt.runtime.proto_tensor import (
            ProtoTensor,  # we need to keep ProtoTensor updated
        )

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
        from brt.runtime.proto_tensor import (
            ProtoTensor,  # we need to keep ProtoTensor updated
        )

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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["start_event"]
        del state["end_event"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)


def register_fabric(fabric_type: str) -> Callable:
    return Registry.register_sub_cls(fabric_type, FabricBase)


def make_fabric(fabric_type: str, kwargs: Dict[str, Any]) -> FabricBase:
    fabric_cls = Registry.get_sub_cls(fabric_type, FabricBase)
    formulated_kwargs = make_kwargs(kwargs)
    return fabric_cls(**formulated_kwargs)
