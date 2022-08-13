# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import inspect
from typing import Callable, Dict, List, Any, Tuple

import os
import numpy as np
import torch
import torch.nn as nn

from brt.runtime import log
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
        if self.capaturing:
            self.history_len = 0
            self.load_histor = None
            self.capacity_history = None
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
        self, loads: np.ndarray, capacities: np.ndarray = None
    ) -> None:
        """
        Capture the flow.
        """
        if not self.capaturing:
            return

        if self.history_len == 0:
            self.load_history = np.zeros_like(loads, dtype=np.float64)
            self.capacity_history = (
                np.zeros_like(capacities, dtype=np.float64)
                if capacities is not None
                else None
            )
        self.load_history = (self.load_history * self.history_len + loads) / (
            self.history_len + 1.0
        )
        if capacities is not None:
            self.capacity_history = (
                self.capacity_history * self.history_len + capacities
            ) / (self.history_len + 1.0)

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

    def skip_routing(self, in_flows, score: torch.Tensor, router_kind: str):
        empty_flows, flows_load = self._check_empty(in_flows)
        if empty_flows:
            if router_kind == "scatter":
                pass
            elif router_kind == "gather":
                pass
            else:
                raise ValueError(f"Unknown router kind: {router_kind}")
        else:
            return False, None

    def _check_empty(self, in_flows) -> Tuple[bool, int]:
        if isinstance(in_flows, torch.Tensor):
            if in_flows.numel() == 0:
                return True, 0
            else:
                return False, 0
        if isinstance(in_flows, (Tuple, List)):
            empty_flows = True
            flows_load = 0
            for flow in in_flows:
                empty_flow, load = self._check_empty(flow)
                empty_flows = empty_flows and empty_flow
                load = np.max([flows_load, load])
                if not empty_flows:
                    return False, load
            return empty_flows, flows_load


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
