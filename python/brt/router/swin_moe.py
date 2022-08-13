# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Any
from brt.router.base import register_router, RouterBase
from brt.router.protocol import make_protocol
from brt.router.fabric import make_fabric


@register_router("swin_moe_scatter")
class SwinMoEScatterRouter(RouterBase):
    def __init__(
        self,
        protocol_type: str = "swin_moe",
        fabric_type: str = "dispatch",
        protocol_kwargs: Dict[str, Any] = None,
        fabric_kwargs: Dict[str, Any] = None,
        capaturing=True,
    ):
        super().__init__(capaturing=capaturing)
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
            "throttling": True,
            "route_logic": ["1d", "2d"],
            "transform": [False, False],
        }
        self.fabric_kwargs.update(built_in_fabric_kwargs)

        if fabric_kwargs is not None:
            self.fabric_kwargs.update(fabric_kwargs)
        self.fabric = make_fabric(self.fabric_type, self.fabric_kwargs)

    def forward(self, tokens, score, logits_wo_noise, logits):
        route_indices, loads, capacities, new_score, loss = self.protocol(
            score, logits_wo_noise, logits
        )

        self.capature_flow_stats(loads, capacities)
        route_indices = self.coordinate_index_format(
            route_indices, loads, self.protocol.index_format, self.fabric.index_format
        )
        in_flows = [tokens, new_score]
        out_flows = self.fabric(in_flows, route_indices, loads, capacities, score)
        return out_flows, loss
