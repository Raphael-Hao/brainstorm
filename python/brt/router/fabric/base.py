# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import os
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from brt.router.utils import make_kwargs
from brt.runtime import log
from brt.runtime.registry import Registry
import brt._C.router as c_router

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

        env_capturing = os.environ.get("BRT_CAPTURE_STATS", "False").lower() in ("true")
        if env_capturing:
            self.capturing = True
            capture_mode = os.environ.get("BRT_CAPTURE_MODE", "avg")
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
                self.captured_fabric_type = captured_fabric_type.split(",")
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

    def forward(
        self, *inputs: Any,
    ):
        raise NotImplementedError("FabricBase is an abstract class for Fabric")

    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        pass

    def capture_flow_stats(self, in_flows, route_indices=None, loads=None):
        if not self.capturing:
            return

        if "dispatch" in self.brt_abs_type and "dispatch" in self.captured_fabric_type:
            self._capture_dispatch_flows(in_flows, route_indices, loads)
        elif "combine" in self.brt_abs_type and "combine" in self.captured_fabric_type:
            self._capture_combine_flows(in_flows)
        else:
            raise ValueError(f"Invalid fabric type: {self.brt_abs_type}")

        self.history_len += 1

    def reset_flow_stats(self):
        self.history_len = 0
        self.ptu_tag_base = 0
        self.load_history = None
        self.ptu_decision_history = None
        self.ptu_grain_history = None
        self.ptu_device_history = None
        self.ptu_dtype_history = None

    def capature(self, mode: str = "cum", fabric_type: str = "dispatch"):
        self.capturing = True
        self.capture_mode = mode
        self.capatured_fabric_type = fabric_type.split(",")

    def stop_capture(self):
        self.capturing = False
        self.capture_mode = None
        self.capatured_fabric_type = []

    def _capture_combine_flows(self, in_flows):

        if len(in_flows) == 0 and isinstance(in_flows, List):
            return

        self._capture_ptu_grains_and_options(in_flows, is_dispatch=False)

        if all(isinstance(flow, List) for flow in in_flows):
            if all(len(flow) > 0 for flow in in_flows):
                in_flows = in_flows[0]
                logger.warning(
                    "Only the first group of in_flows is captured, please make sure the loads are the same for all groups."
                )
            else:
                return

        self._capture_load_from_flows(in_flows)

    def _capture_dispatch_flows(self, in_flows, route_indices, loads):
        self._capture_correlations(route_indices, loads)
        self._capture_ptu_grains_and_options(in_flows, is_dispatch=True)
        self._capture_laod_from_loads(loads)

    def _capture_load_from_flows(self, in_flows: List[torch.Tensor]) -> None:
        path_num = len(in_flows)

        if self.history_len == 0:
            self.load_history = np.zeros(path_num, dtype=np.float64)

        current_load = np.array([flow.size(0) for flow in in_flows], dtype=np.float64)
        if self.capture_mode == "avg":
            self.load_history = (
                self.load_history * self.history_len + current_load
            ) / (self.history_len + 1.0)

        elif self.capture_mode == "max":
            self.load_history = np.maximum(self.load_history, current_load)

        elif self.capture_mode == "cum":
            self.load_history = self.load_history + current_load

    def _capture_laod_from_loads(self, loads: torch.Tensor):
        loads_np = loads.cpu().numpy()
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

    def _capture_ptu_grains_and_options(self, flows, is_dispatch=True) -> None:
        """
        Capture the flow shape.
        """
        flows = self._listing_flows(flows, is_dispatch)

        if self.ptu_grain_history is None:
            if is_dispatch:
                self.ptu_grain_history = [flow.shape for flow in flows]
                self.ptu_dtype_history = [flow.dtype for flow in flows]
                self.ptu_device_history = [flow.device for flow in flows]
            else:
                self.ptu_grain_history = [flow[0].shape for flow in flows]
                self.ptu_dtype_history = [flow[0].dtype for flow in flows]
                self.ptu_device_history = [flow[0].device for flow in flows]

    def _capture_correlations(self, indices: torch.Tensor, loads: torch.Tensor):
        assert hasattr(
            self, "protocol"
        ), "Correlation capturing is only supported for Router with protocol!"
        path_num = len(loads)
        tag_indices = indices
        if not self.is_tag_index:
            tag_indices = c_router.convert_index_format(indices, loads, True)

        current_ptu_path = [
            tag_indices[: loads[i], i].cpu().numpy() + self.ptu_tag_base
            for i in range(path_num)
        ]
        current_ptu_path = [
            current_ptu_path[i][current_ptu_path[i] != 0] + self.ptu_tag_base
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

        for history in self.ptu_decision_history:
            self.ptu_tag_base = history.max()

    def _listing_flows(self, flows, is_dispatch=True):
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


def register_fabric(fabric_type: str) -> Callable:
    return Registry.register_sub_cls(fabric_type, FabricBase)


def make_fabric(fabric_type: str, kwargs: Dict[str, Any]) -> FabricBase:
    fabric_cls = Registry.get_sub_cls(fabric_type, FabricBase)
    formulated_kwargs = make_kwargs(kwargs)
    return fabric_cls(**formulated_kwargs)


def switch_capture(m: nn.Module, capture=True, mode="cum", fabric_type="fabric"):
    for child_m in m.children():
        switch_capture(child_m, capture=capture, mode=mode, fabric_type=fabric_type)
    if isinstance(m, FabricBase):
        if capture:
            m.capature(mode=mode, fabric_type=fabric_type)
        else:
            m.stop_capture()
    return m


def reset_router_stats(m: nn.Module):
    for child_m in m.children():
        reset_router_stats(child_m)
    if isinstance(m, FabricBase):
        m.reset_flow_stats()
    return m
