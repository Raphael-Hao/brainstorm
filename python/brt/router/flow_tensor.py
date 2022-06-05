from typing import List, Tuple

import torch
from brt.common import log

logger = log.get_logger(__file__)


class FlowTensor(torch.Tensor):
    CHECK_TAGS = False

    def init_flow(self):
        if not self.flow_initilized:
            self.tags = []
            self.loads = []

    @property
    def flow_initilized(self):
        return hasattr(self, "tags") and hasattr(self, "loads")

    @property
    def tag(self) -> torch.Tensor:
        if self.flow_empty():
            logger.warning(
                "Trying to get tag from an FlowTensor without packed tag and load"
            )
            return torch.zeros(0, 1, dtype=torch.int64, device=self.device)
        return self.tags[-1]

    @property
    def load(self) -> int:
        if self.flow_empty():
            logger.warning(
                "Trying to get load from an FlowTensor without packed tag and load"
            )
            return 0
        return self.loads[-1]

    def pack(self, tag: torch.Tensor, load: int):
        self.init_flow()
        assert isinstance(tag, torch.Tensor), "tag must be a torch.Tensor for packing"
        self.tags.append(tag)
        self.loads.append(load)
        return self

    def unpack(self):
        if not hasattr(self, "tags") or not hasattr(self, "loads"):
            logger.error("Unpacking a not initialized FlowTensor")
        tag = self.tags.pop()
        load = self.loads.pop()
        return self, tag, load

    def deep_pack(self, tags: List[torch.Tensor], loads: List[int]):
        self.tags = tags
        self.loads = loads
        return self

    def deep_unpack(self):
        if not hasattr(self, "tags") or not hasattr(self, "loads"):
            logger.error("Unpacking a not initialized FlowTensor")
        tags, self.tags = self.tags, []
        loads, self.loads = self.loads, []
        return self, tags, loads

    @property
    def pack_size(self) -> int:
        return len(self.tags)

    def flow_empty(self):
        tags = getattr(self, "tags", [])
        loads = getattr(self, "loads", [])
        if not tags or not loads:
            return True
        return False

    def __repr__(self):
        return f"FlowTensor:\ndata: {super().__repr__()}\ncurrent tag: {self.tag}\ncurrent load: {self.load}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        all_tags, all_loads = collect_tags_loads(args)
        assert len(all_tags) > 0 and len(all_loads) > 0
        if cls.CHECK_TAGS:
            for tags in all_tags:
                for i in range(len(tags)):
                    assert torch.allclose(all_tags[0][i], tags[i])
        ret = super().__torch_function__(func, types, args, kwargs)
        pack_ret(ret, all_tags[0], all_loads[0])

        return ret


def collect_tags_loads(args):
    all_tags = []
    all_loads = []

    if isinstance(args, FlowTensor):
        all_tags.append(args.tags)
        all_loads.append(args.loads)

    if isinstance(args, (Tuple, List)):
        for a in args:
            _all_tags, _all_loads = collect_tags_loads(a)
            all_tags.extend(_all_tags)
            all_loads.extend(_all_loads)

    return all_tags, all_loads


def pack_ret(ret, tags, loads):

    if isinstance(ret, FlowTensor):
        ret = ret.deep_pack(tags, loads)

    if isinstance(ret, (Tuple, List)):
        ret = type(ret)(pack_ret(t) for t in ret)

    return ret


def init_flow_tensor(
    data: torch.Tensor, tags: List[torch.Tensor], loads: List[int]
) -> FlowTensor:
    data = data.as_subclass(FlowTensor)
    data.deep_pack(tags, loads)
    return data


def deinit_flow_tensor(
    data: FlowTensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    data, tags, loads = data.deep_unpack()
    data = data.as_subclass(torch.Tensor)
    return data, tags, loads
