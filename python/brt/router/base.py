# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import inspect
from typing import Callable, Dict, List, Any

import os
import torch
import torch.nn as nn
import numpy as np

from brt.runtime import log, Registry
from brt.router.utils import convert_index_format, make_kwargs
from brt.trace.init_arg import trace_init

__all__ = ["RouterBase"]

logger = log.get_logger(__file__)


class RouterBase(nn.Module):
    def __init__(self, capturing=False, capture_mode="cum"):
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
            captured_fabric_type = os.environ.get("BRT_CAPTURED_FABRIC_TYPE")
            if captured_fabric_type is not None:
                self.captured_fabric_type = captured_fabric_type.split(":")
            else:
                self.captured_fabric_type = ["dispatch"]
        else:
            self.capturing = False

        self.history_len = 0
        self.load_history: np.ndarray = None
        self.ptu_decision_history: List[np.ndarray] = None
        self.ptu_tag_base = 0
        self.ptu_grain_history: List[torch.Size] = None
        self.ptu_dtype_history: List[torch.dtype] = None
        self.ptu_device_history: List[torch.device] = None

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
        self.ptu_tag_base = 0
        self.load_history = None
        self.ptu_decision_history = None
        self.ptu_grain_history = None
        self.ptu_device_history = None
        self.ptu_dtype_history = None

    def capture(self, mode=True):
        """Switch the capturing mode OFF or ON

        Args:
            mode (bool, optional): Defaults to True.
        """
        self.capturing = mode

    def capture_flow_stats(
        self,
        fabric_type: str,
        in_flows: List[torch.Tensor],
        route_indices: torch.Tensor = None,
        loads: torch.Tensor = None,
    ) -> None:
        """
        Capture the flow stats.
        """
        if not self.capturing:
            return

        if "dispatch" in fabric_type and "dispatch" in self.captured_fabric_type:
            self.capture_dispatch_flows(in_flows, route_indices, loads)
        elif "combine" in fabric_type and "combine" in self.captured_fabric_type:
            self.capture_combine_flows(in_flows)
        else:
            return

        self.history_len += 1

    def capture_combine_flows(self, in_flows):

        if len(in_flows) == 0 and isinstance(in_flows, List):
            return

        self.capture_ptu_grains_and_options(in_flows, is_dispatch=False)

        if all(isinstance(flow, List) for flow in in_flows):
            if all(len(flow) > 0 for flow in in_flows):
                in_flows = in_flows[0]
                logger.warning(
                    "Only the first group of in_flows is captured, plz make sure the loads are the same for all groups."
                )
            else:
                return

        self.capture_load_from_flows(in_flows)

    def capture_dispatch_flows(self, in_flows, route_indices, loads):
        self.capture_correlations(route_indices, loads)
        self.capture_ptu_grains_and_options(in_flows, is_dispatch=True)
        self.capture_laod_from_protocol(loads)

    def capture_load_from_flows(self, in_flows: List[torch.Tensor]) -> None:
        path_num = len(in_flows)

        if self.history_len == 0:
            self.load_history = np.zeros(path_num, dtype=np.float64)

        current_load = np.array(
            [flow.size(0) for flow in in_flows], dtype=np.float64
        )
        if self.capture_mode == "avg":
            self.load_history = (
                self.load_history * self.history_len + current_load
            ) / (self.history_len + 1.0)

        elif self.capture_mode == "max":
            self.load_history = np.maximum(self.load_history, current_load)

        elif self.capture_mode == "cum":
            self.load_history = self.load_history + current_load

    def capture_laod_from_protocol(self, loads: torch.Tensor):

        loads_np = loads.numpy()
        if self.history_len == 0:
            self.load_history = np.zeros_like(loads_np, dtype=np.float64)

        if self.capture_mode == "avg":
            self.load_history = (self.load_history * self.history_len + loads_np) / (
                self.history_len + 1.0
            )

        elif self.capture_mode == "max":
            self.load_history = np.maximum(self.load_history, loads_np)

        elif self.capture_mode == "cum":
            self.load_history = self.load_history + loads_np

    def capture_ptu_grains_and_options(self, flows, is_dispatch=True) -> None:
        """
        Capture the flow shape.
        """
        flows = self.listing_flows(flows, is_dispatch)

        if self.check_ptu_consistency(flows, is_dispatch):
            if self.ptu_grain_history is None:
                if is_dispatch:
                    self.ptu_grain_history = [flow.shape for flow in flows]
                    self.ptu_dtype_history = [flow.dtype for flow in flows]
                    self.ptu_device_history = [flow.device for flow in flows]
                else:
                    self.ptu_grain_history = [flow[0].shape for flow in flows]
                    self.ptu_dtype_history = [flow[0].dtype for flow in flows]
                    self.ptu_device_history = [flow[0].device for flow in flows]
        else:
            self.ptu_grain_history = None

    def capture_correlations(self, indices: torch.Tensor, loads: torch.Tensor):
        assert hasattr(
            self, "protocol"
        ), "Correlation capturing is only supported for Router with protocol!"
        path_num = len(loads)
        src_indices = convert_index_format(
            indices, loads, self.protocol.index_format, "src_index"
        )
        current_ptu_path = [
            src_indices[: loads[i], i].cpu().numpy() + self.ptu_tag_base
            for i in range(path_num)
        ]
        if self.history_len == 0:
            self.ptu_decision_history = current_ptu_path
        else:
            self.ptu_decision_history = [
                np.concatenate(
                    (self.ptu_decision_history[i], current_ptu_path[i]), axis=None
                )
                for i in range(path_num)
            ]
        self.ptu_tag_base += loads.sum().item()

    def listing_flows(self, flows, is_dispatch=True):
        if is_dispatch:
            if isinstance(flows, torch.Tensor):
                return [flows]
            return flows

        if isinstance(flows, List):
            if isinstance(flows[0], torch.Tensor):
                return [flows]
            return flows

    def check_ptu_consistency(self, flows, is_dispatch=True) -> bool:
        if self.ptu_grain_history is None:
            if self.history_len == 0:
                return True
            else:
                return False
        if is_dispatch:
            for flow_id, flow in enumerate(flows):
                if (
                    flow.shape[1:] != self.ptu_grain_history[flow_id][1:]
                    or flow.dtype != self.ptu_dtype_history[flow_id]
                    or flow.device != self.ptu_device_history[flow_id]
                ):
                    return False
        else:
            for flow_id, flow in enumerate(flows):
                if (
                    flow[0].shape[1:] != self.ptu_grain_history[flow_id][1:]
                    or flow[0].dtype != self.ptu_dtype_history[flow_id]
                    or flow[0].device != self.ptu_device_history[flow_id]
                ):
                    return False

        return True

    def placement(self, route_indices, loads, capacities):
        route_indices = route_indices[:, self.placement_index]
        loads = loads[:, self.placement_index]
        capacities = capacities[:, self.placement_index]
        return route_indices, loads, capacities


def register_router(router_type: str) -> Callable:
    global_register_func = Registry.register_sub_cls(router_type, RouterBase)

    def local_register_func(router_cls):

        if not issubclass(router_cls, RouterBase):
            raise ValueError(f"{router_cls} is not a subclass of RouterBase")

        router_cls = trace_init(router_cls)

        router_cls._router_type = router_type

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


def switch_router_mode(m: nn.Module, capture=True):
    for child_m in m.children():
        switch_router_mode(child_m, capture=capture)
    if isinstance(m, RouterBase):
        m.capture(capture)
    return m


def reset_router_stats(m: nn.Module):
    for child_m in m.children():
        reset_router_stats(child_m)
    if isinstance(m, RouterBase):
        m.reset_flow_stats()
    return m
