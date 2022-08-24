# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Any

from brt.runtime import log
from brt.router.base import RouterBase, register_router
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.runtime.proto_tensor import ProtoTensor

__all__ = [
    "LoopRouter",
]

logger = log.get_logger(__file__)

@register_router("loop")
class LoopRouter(RouterBase):
    def __init__(
        self,
        netlet,
        protocol_type: str = "topk",
        dispatch_fabric_type: str = "dispatch",
        combine_fabric_type: str = "combine",
        protocol_kwargs: Dict[str, Any] = None,
        dispatch_fabric_kwargs: Dict[str, Any] = None,
        combine_fabric_kwargs: Dict[str, Any] = None,
        capturing: bool = False,
        capture_mode: str = "cum",
    ):
        super().__init__(capturing=capturing, capture_mode=capture_mode)
        self.protocol_type = protocol_type
        self.dispatch_fabric_type = dispatch_fabric_type
        self.combine_fabric_type = combine_fabric_type
        self.protocol_kwargs = protocol_kwargs
        self.dispatch_fabric_kwargs = dispatch_fabric_kwargs
        self.combine_fabric_kwargs = combine_fabric_kwargs
        self.protocol = make_protocol(self.protocol_type, self.protocol_kwargs)
        self.dispatch_fabric = make_fabric("dispatch", self.dispatch_fabric_kwargs)
        self.combine_fabric = make_fabric("combine", self.dispatch_fabric)
        self.netlet = netlet

    def forward(self, in_flow: ProtoTensor) -> ProtoTensor:
        target_flow = in_flow
        pending_combine_flows = []
        while target_flow.numel() > 0:
            target_flow, score = self.netlet(target_flow)
            route_indices, loads, capacities = self.protocol(score)
            route_indices = self.coordinate_index_format(
                route_indices,
                loads,
                self.protocol.index_format,
                self.dispatch_fabric.index_format,
            )
            self.capature_flow_stats(loads, capacities)
            dispatched_flows = self.dispatch_fabric(
                target_flow, route_indices, loads, capacities, score
            )
            target_flow, completed_flow = dispatched_flows
            pending_combine_flows.append(completed_flow)
        out_flows = self.combine_fabric(pending_combine_flows)
        return out_flows

    def update_protocol(self, **kwargs):
        self.protocol.update(**kwargs)
