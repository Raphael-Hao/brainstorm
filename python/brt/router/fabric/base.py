# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, Any

import torch
import torch.nn as nn
from brt.runtime import log
from brt.router.utils import make_kwargs
from brt.runtime.registry import Registry

logger = log.get_logger(__file__)


class FabricBase(nn.Module):
    def __init__(self, flow_num: int, index_format: str = None, **kwargs) -> None:
        super().__init__()
        self.flow_num = flow_num
        assert index_format in [
            "tag_index",
            "seat_index",
        ], f"index_format should be dst_index, src_index, but got {index_format}"

        self.is_tag_index = index_format == "tag_index"
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start_timer(self):
        self.start_event.record(torch.cuda.current_stream())

    def end_timer(self, timer_name):
        self.end_event.record(torch.cuda.current_stream())
        self.end_event.synchronize()
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["start_event"]
        del state["end_event"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    # @classmethod
    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        pass


def register_fabric(fabric_type: str) -> Callable:
    return Registry.register_sub_cls(fabric_type, FabricBase)


def make_fabric(fabric_type: str, kwargs: Dict[str, Any]) -> FabricBase:
    fabric_cls = Registry.get_sub_cls(fabric_type, FabricBase)
    formulated_kwargs = make_kwargs(kwargs)
    return fabric_cls(**formulated_kwargs)
