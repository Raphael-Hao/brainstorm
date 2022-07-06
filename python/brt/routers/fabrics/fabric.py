# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, List, Tuple

import torch
from brt.common import log
from brt.frontend import nn

from ..proto_tensor import deinit_proto_tensor, init_proto_tensor

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


class FabricFactory:
    registry: Dict[str, SwitchFabric] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def register_func(fabric_cls: SwitchFabric):
            if name in cls.registry:
                logger.warning(f"Fabric: {name} is already registered, overwrite it")
            fabric_cls.forward = torch.jit.ignore(fabric_cls.forward)
            cls.registry[name] = fabric_cls
            return fabric_cls

        return register_func

    @classmethod
    def make_fabric(cls, fabric_type, **kwargs) -> SwitchFabric:
        for key, value in kwargs.items():
            logger.debug(f"{key}: {value}")

        if fabric_type not in cls.registry:
            logger.error(f"Fabric: {fabric_type} is not registered")
            return None
        fabric_cls = cls.registry[fabric_type]
        fab = fabric_cls(**kwargs)

        return fab
