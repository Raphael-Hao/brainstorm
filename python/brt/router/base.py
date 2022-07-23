# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import inspect
from typing import Callable, Dict, List, Type, Union, Any

import os
import torch
import torch.nn as nn

from brt.common import log
from brt.runtime import Registry
from brt.router.utils import convert_index_format, make_kwargs
from brt.trace.initialize import trace_init

__all__ = ["RouterBase"]

logger = log.get_logger(__file__)


class RouterBase(nn.Module):
    def __init__(self, capaturing=False):
        super().__init__()
        env_capaturing = os.environ.get("BRT_CAPTURE_STATS", "False").lower() in (
            "true"
        )
        if env_capaturing or capaturing:
            self.capaturing = True
        self.history_len = 0
        self.register_parameter("load_history", None)
        self.register_parameter("capacity_history", None)
        self.schedule_functions: List[Callable] = []

    def forward(self):
        self.run_schedule()

    def coordinate_index_format(
        self,
        route_indices: torch.Tensor,
        loads: torch.Tensor,
        protocol_index_fmt: str,
        fabric_index_fmt: str,
    ):
        """
        Coordinate the route index format between protocol and fabric.
        """
        new_route_indices = convert_index_format(
            route_indices, loads, protocol_index_fmt, fabric_index_fmt
        )
        return new_route_indices

    def capature_flow_stats(
        self, loads: torch.Tensor, capacities: torch.Tensor = None
    ) -> None:
        """
        Capture the flow.
        """
        if not self.capaturing:
            return

        with torch.no_grad():
            if self.history_len == 0:
                self.load_history = torch.zeros_like(loads)
                self.capacity_history = (
                    torch.zeros_like(capacities) if capacities is not None else None
                )
            self.load_history = (self.load_history * self.history_len + loads) / (
                self.history_len + 1
            )
            if capacities is not None:
                self.capacity_history = (
                    self.capacity_history * self.history_len + capacities
                ) / (self.history_len + 1)

    def reset_flow_stats(self):
        self.history_len = 0
        self.load_history = None
        self.capacity_history = None

    def inject_schedule(self, schedule_function):
        self.schedule_functions.append(schedule_function)

    def run_schedule(self):
        for func in self.schedule_functions:
            func()


def register_router(router_type: str) -> Callable:
    global_register_func = Registry.register_sub_cls(router_type, RouterBase)

    def local_register_func(router_cls):

        if not issubclass(router_cls, RouterBase):
            raise ValueError(f"{router_cls} is not a subclass of RouterBase")

        router_cls = trace_init(router_cls)

        return global_register_func(router_cls)

    return local_register_func


def make_router(router_type: str, kwargs: Dict[str, Any]) -> RouterBase:
    router_cls = Registry.get_sub_cls(router_type, RouterBase)
    if router_cls is None:
        raise ValueError(f"Router type: {router_type} is not registered.")
    formulated_kwargs = make_kwargs(kwargs)
    return router_cls(**formulated_kwargs)


def is_router(cls_or_instance) -> bool:

    if not inspect.isclass(cls_or_instance):
        router_cls = cls_or_instance.__class__
    else:
        router_cls = cls_or_instance

    return Registry.sub_cls_exists_and_registered(router_cls, RouterBase)
