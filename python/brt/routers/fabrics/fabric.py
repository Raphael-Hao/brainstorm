# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, List, Tuple, Type

import torch
import torch.nn as nn
from brt.common import log

from ..proto_tensor import deinit_proto_tensor, init_proto_tensor

logger = log.get_logger(__file__)


class FabricBase(nn.Module):
    def __init__(self, path_num: int) -> None:
        super().__init__()
        self.path_num = path_num
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

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

    def __getstate__(self):
        state = self.__dict__.copy()
        print(state)
        del state["start_event"]
        del state["end_event"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)


class FabricFactory:
    registry: Dict[str, FabricBase] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def register_func(fabric_cls) -> Type[FabricBase]:
            if name in cls.registry:
                logger.warning(f"Fabric: {name} is already registered, overwrite it")
            if not issubclass(fabric_cls, FabricBase):
                raise ValueError(f"Fabric: {name} is not a subclass of FabricBase")
            fabric_cls.forward = torch.jit.ignore(fabric_cls.forward)
            cls.registry[name] = fabric_cls
            return fabric_cls

        return register_func

    @classmethod
    def make_fabric(cls, fabric_type, **kwargs) -> FabricBase:
        for key, value in kwargs.items():
            logger.debug(f"{key}: {value}")

        if fabric_type not in cls.registry:
            raise ValueError(f"Fabric: {fabric_type} is not registered")
        fabric_cls = cls.registry[fabric_type]
        fabric_inst = fabric_cls(**kwargs)

        return fabric_inst
