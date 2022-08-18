# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import inspect
from typing import Callable, Dict, List, Any

import os
import torch
import torch.nn as nn

from brt.runtime import log, Registry
from brt.router.utils import convert_index_format, make_kwargs
from brt.trace.initialize import trace_init

__all__ = ["RouterBase"]

logger = log.get_logger(__file__)


class RouterBase(nn.Module):
    def __init__(self, capturing=False, capture_mode="a"):
        super().__init__()
        env_capturing = os.environ.get("BRT_CAPTURE_STATS", "False").lower() in ("true")
        if env_capturing or capturing:
            self.capturing = True
            self.capture_mode = capture_mode
        else:
            self.capturing = False
        if self.capturing:
            self.history_len = 0
            self.register_buffer("load_history", None)
            self.register_buffer("capacity_history", None)
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

    def capture_flow_stats(
        self, loads: torch.Tensor, capacities: torch.Tensor = None
    ) -> None:
        """
        Capture the flow.
        """
        if not self.capturing:
            return

        if self.history_len == 0:
            self.load_history = torch.zeros_like(
                loads, dtype=torch.float64, device="cpu"
            )
            self.capacity_history = (
                torch.zeros_like(capacities, dtype=torch.float64, device="cpu")
                if capacities is not None
                else None
            )

        if self.capture_mode == "a":
            self.load_history = (self.load_history * self.history_len + loads) / (
                self.history_len + 1.0
            )
            if capacities is not None:
                self.capacity_history = (
                    self.capacity_history * self.history_len + capacities
                ) / (self.history_len + 1.0)
        elif self.capture_mode == "m":
            self.load_history = torch.maximum(self.load_history, loads)
            if capacities is not None:
                self.capacity_history = torch.maximum(self.capacity_history, capacities)
        elif self.capture_mode == "c":
            self.load_history = self.load_history + loads
            if capacities is not None:
                self.capacity_history = self.capacity_history + capacities

        self.history_len += 1

    def reset_flow_stats(self):
        self.history_len = 0
        self.load_history = None
        self.capacity_history = None

    def inject_schedule(self, schedule_function):
        self.schedule_functions.append(schedule_function)

    def capature_flow_shape(self, flows) -> None:
        pass

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
