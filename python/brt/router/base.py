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
    def __init__(self, capturing=False, capture_mode="avg"):
        super().__init__()
        env_capturing = os.environ.get("BRT_CAPTURE_STATS", "False").lower() in ("true")
        if env_capturing or capturing:
            self.capturing = True
            assert capture_mode in (
                "avg",
                "max",
                "cum",
            ), f"Invalid capture mode: {capture_mode},valid options:"
            "avg for average"
            "max for maximum"
            "cum for cumulative"
            self.capture_mode = capture_mode
        else:
            self.capturing = False
        if self.capturing:
            self.history_len = 0
            self.register_buffer("load_history", None)
            self.register_buffer("capacity_history", None)
            self.granularities: List[torch.Size] = None

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
        if protocol_index_fmt is None:
            assert (
                fabric_index_fmt is None
            ), "The used fabric is not compatible with the protocol."
            return route_indices
        new_route_indices = convert_index_format(
            route_indices, loads, protocol_index_fmt, fabric_index_fmt
        )
        return new_route_indices

    def reset_flow_stats(self):
        self.history_len = 0
        self.load_history = None
        self.capacity_history = None
        self.granularities = None

    def capture_flow_stats(
        self,
        fabric_type: str,
        in_flows: List[torch.Tensor],
        loads: torch.Tensor = None,
        capacities: torch.Tensor = None,
    ) -> None:
        """
        Capture the flow stats.
        """
        if not self.capturing:
            return

        if "dispatch" in fabric_type:
            self.capture_outbound_flows(in_flows, loads, capacities)
        elif "combine" in fabric_type:
            self.capture_inbound_flows(in_flows)
        else:
            return

        self.history_len += 1

    def capture_inbound_flows(self, in_flows):
        if len(in_flows) == 0 and isinstance(in_flows, List):
            return

        if all(isinstance(flow, List) for flow in in_flows):
            if all(len(flow) > 0 for flow in in_flows):
                in_flows = in_flows[0]
                logger.warning(
                    "Only the first group of in_flows is captured, plz make sure the loads are the same for all groups."
                )
            else:
                print(in_flows)
                return

        self.capture_load_from_flows(in_flows)
        self.capture_shape(in_flows)

    def capture_outbound_flows(self, in_flows, loads, capacities):
        self.capture_shape(in_flows)
        self.capture_laod_from_protocol(loads, capacities)

    def capture_load_from_flows(self, in_flows: List[torch.Tensor]) -> None:
        path_num = len(in_flows)

        if self.history_len == 0:
            self.load_history = torch.zeros(path_num, dtype=torch.float64, device="cpu")

        current_load = torch.tensor(
            [flow.size(0) for flow in in_flows], dtype=torch.float64, device="cpu"
        )
        if self.capture_mode == "avg":
            self.load_history = (
                self.load_history * self.history_len + current_load
            ) / (self.history_len + 1.0)

        elif self.capture_mode == "max":
            self.load_history = torch.maximum(self.load_history, current_load)

        elif self.capture_mode == "cum":
            self.load_history = self.load_history + current_load

    def capture_laod_from_protocol(self, loads, capacities):

        if self.history_len == 0:
            self.load_history = torch.zeros_like(
                loads, dtype=torch.float64, device="cpu"
            )
            self.capacity_history = (
                torch.zeros_like(capacities, dtype=torch.float64, device="cpu")
                if capacities is not None
                else None
            )

        if self.capture_mode == "avg":
            self.load_history = (self.load_history * self.history_len + loads) / (
                self.history_len + 1.0
            )

            if capacities is not None:
                self.capacity_history = (
                    self.capacity_history * self.history_len + capacities
                ) / (self.history_len + 1.0)

        elif self.capture_mode == "max":
            self.load_history = torch.maximum(self.load_history, loads)

            if capacities is not None:
                self.capacity_history = torch.maximum(self.capacity_history, capacities)

        elif self.capture_mode == "cum":
            self.load_history = self.load_history + loads

            if capacities is not None:
                self.capacity_history = self.capacity_history + capacities

    def capture_shape(self, flows) -> None:
        """
        Capture the flow shape.
        """
        if self.check_granularity_consistency(flows):
            if self.granularities is None:
                self.granularities = [flow.shape for flow in flows]
        else:
            self.granularities = None

    def capture(self, mode=True):
        """Switch the capturing mode OFF or ON

        Args:
            mode (bool, optional): Defaults to True.
        """
        self.capturing = mode

    def check_granularity_consistency(self, flows) -> bool:
        if self.granularities is None:
            if self.history_len == 0:
                return True
            else:
                return False

        for flow, granularity in zip(flows, self.granularities):
            if flow.shape[1:] != granularity[1:]:
                return False

        return True

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


def router_capture(m: nn.Module, mode=True):
    for child_m in m.children():
        router_capture(child_m)
    if isinstance(m, RouterBase):
        m.capture(mode)
    return m
