from collections.abc import Iterable
from typing import List

import torch


class FlowTensor(torch.Tensor):
    CHECK_TAGS = False

    def init_flow(self, tag: torch.Tensor, load):
        self.tag = tag
        self.load = load
        return self

    def __repr__(self):
        return f"FlowTensor:\ndata: {super().__repr__()}\ntag: {self.tag}\nload: {self.load}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        tags = []
        loads = []
        for _, arg in enumerate(args):
            if isinstance(arg, FlowTensor):
                tags.append(arg.tag)
                loads.append(arg.load)
                continue
            if isinstance(arg, torch.Tensor):
                continue
            if isinstance(arg, Iterable):
                for t in arg:
                    if isinstance(t, FlowTensor):
                        tags.append(t.tag)
                        loads.append(t.load)
        assert len(tags) > 0 and len(loads) > 0
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, torch.Tensor):
            ret.tag = tags[0]
            ret.load = loads[0]
            return ret
        if isinstance(ret, Iterable):
            for t in ret:
                if isinstance(t, torch.Tensor):
                    t.tag = tags[0]
                    t.load = loads[0]
        # if cls.CHECK_TAGS:
        #     for tag in tags:
        #         assert torch.allclose(tags[0], tag)
        return ret


def init_flow_tensor(data: torch.Tensor, tag: torch.Tensor, load: int) -> FlowTensor:
    data.__class__ = FlowTensor
    data.init_flow(tag, load)
    return data


def deinit_flow_tensor(data: FlowTensor, tag: torch.Tensor, load: int) -> torch.Tensor:
    data.__class__ = torch.Tensor
    return data
