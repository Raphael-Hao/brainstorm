# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Any

import torch
from brt.runtime import log
from brt.router.base import RouterBase, register_router
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol

__all__ = [
    "ScatterRouter",
    "SwinMoEScatterRouter",
]

logger = log.get_logger(__file__)


@register_router("scatter")
class ScatterRouter(RouterBase):
    def __init__(
        self,
        dispatch_score=False,
        protocol_type: str = "topk",
        fabric_type: str = "dispatch",
        protocol_kwargs: Dict[str, Any] = None,
        fabric_kwargs: Dict[str, Any] = None,
        throttling=False,
        capturing=False,
        capture_mode: str = "cum",
    ):
        """base scatter router

        Args:
            protocol_type (str, optional): protocol type. Defaults to "topk".
                topk: select the topk of gate results as the route destinations
                threshold: select the gate results that are larger than threshold as the route destinations
                supported keyword args for protocol:
                    k (int): k only for topk protocol, default to 1
                    threshold (float): threshold for threshold protocol, default to 0
                    residual_path (int, optinal): a path that is used to directly route residual flows

            fabric_type (str, optional): fabric type. Defaults to "dispatch".
                dispatch: dispatch source flows to destinations
                homo_dispatch: dispatch source flows to destinations in the form of a fused ProtoTensor.
                supported keyword args for fabric:
                    route_logic (str, optional): route logic. Defaults to "1d" only for dispatch.
                        1d: route along the 1st dimension, selecting data from a tensor with shape (batch_size, ...)
                        2d: route along the first 2 dimensions, selecting data from a tensor with shape (batch_size, path_num, ...)
                    transform (bool, optional): whether to transform the route result to the original shape. Defaults to False.
            Capturing (bool, optional): whether to capture the flow stats. Defaults to True.
        """
        super().__init__(capturing=capturing, capture_mode=capture_mode)
        self.dispatch_score = dispatch_score

        self.protocol_type = protocol_type

        self.protocol_kwargs = {"index_format": "src_index", "index_gen_opt": True}

        if self.protocol_type == "topk":
            built_in_protocol_kwargs = {
                "top_k": 1,
                "supported_capacities": None,
            }
        elif self.protocol_type == "threshold":
            built_in_protocol_kwargs = {
                "threshold": 0.0,
                "residual_path": -1,
                "supported_capacities": None,
            }
        else:
            built_in_protocol_kwargs = {}
        self.protocol_kwargs.update(built_in_protocol_kwargs)

        if protocol_kwargs is not None:
            self.protocol_kwargs.update(protocol_kwargs)

        self.protocol = make_protocol(protocol_type, self.protocol_kwargs)

        self.fabric_type = fabric_type

        self.fabric_kwargs = {}

        if "dispatch" in self.fabric_type:
            built_in_fabric_kwargs = {
                "flow_num": 1,
                "route_logic": "1d",
                "transform": False,
            }
            if self.dispatch_score:
                built_in_fabric_kwargs["route_logic"] = ["1d", "2d"]
                built_in_fabric_kwargs["transform"] = [False, False]
                built_in_fabric_kwargs["flow_num"] = 2

        self.fabric_kwargs.update(built_in_fabric_kwargs)

        if fabric_kwargs is not None:
            self.fabric_kwargs.update(fabric_kwargs)

        self.fabric = make_fabric(fabric_type, self.fabric_kwargs)

        self.throttling = throttling

    def forward(self, in_flows, score: torch.Tensor):
        route_indices, loads, capacities = self.protocol(score)

        if self.dispatch_score:
            if isinstance(in_flows, List):
                in_flows = in_flows.append(score)
            else:
                in_flows = [in_flows, score]

        route_indices = self.coordinate_index_format(
            route_indices, loads, self.protocol.index_format, self.fabric.index_format
        )

        if self.throttling:
            real_loads = torch.minimum(loads, capacities)
        else:
            real_loads = loads

        self.capture_flow_stats(self.fabric_type, in_flows, route_indices, loads)

        out_flows = self.fabric(in_flows, route_indices, real_loads, score)
        return out_flows

    def update_protocol(self, **kwargs):
        self.protocol.update(**kwargs)


@register_router("swin_moe_scatter")
class SwinMoEScatterRouter(RouterBase):
    ALLOWED_PROTOCOL_TYPES = ["swin_moe"]

    def __init__(
        self,
        protocol_type: str = "swin_moe",
        fabric_type: str = "dispatch",
        protocol_kwargs: Dict[str, Any] = None,
        fabric_kwargs: Dict[str, Any] = None,
        throttling=False,
        capturing=True,
        capture_mode: str = "cum",
    ):
        super().__init__(capturing=capturing, capture_mode=capture_mode)
        assert (
            protocol_type in self.ALLOWED_PROTOCOL_TYPES
        ), f"protocol_type {protocol_type} is not supported by SwinMoEScatterRouter"
        self.protocol_type = protocol_type
        self.protocol_kwargs = {"index_format": "dst_index", "index_gen_opt": True}
        built_in_protocol_kwargs = {
            "top_k": 1,
            "capacity_factor": 0,
            "gate_noise": 0.0,
            "group": None,
            "is_postscore": True,
            "batch_prioritized_routing": False,
            "normalize_gate": True,
            "is_gshard_loss": True,
            "alignment": 1,
        }
        self.protocol_kwargs.update(built_in_protocol_kwargs)
        if protocol_kwargs is not None:
            self.protocol_kwargs.update(protocol_kwargs)
        self.protocol = make_protocol(self.protocol_type, self.protocol_kwargs)

        self.fabric_type = fabric_type
        self.fabric_kwargs = {}
        built_in_fabric_kwargs = {
            "flow_num": 2,
            "route_logic": ["1d", "2d"],
            "transform": [False, False],
        }
        self.fabric_kwargs.update(built_in_fabric_kwargs)

        if fabric_kwargs is not None:
            self.fabric_kwargs.update(fabric_kwargs)
        self.fabric = make_fabric(self.fabric_type, self.fabric_kwargs)

        self.throttling = throttling

    def forward(self, in_flows, score, logits_wo_noise, logits):
        route_indices, loads, capacities, new_score, loss = self.protocol(
            score, logits_wo_noise, logits
        )

        if isinstance(in_flows, List):
            in_flows = in_flows.append(new_score)
        else:
            in_flows = [in_flows, new_score]

        route_indices = self.coordinate_index_format(
            route_indices, loads, self.protocol.index_format, self.fabric.index_format
        )
        self.capture_flow_stats(self.fabric_type, in_flows, loads, capacities)
        out_flows = self.fabric(in_flows, route_indices, loads, capacities, score)
        return out_flows, loss
