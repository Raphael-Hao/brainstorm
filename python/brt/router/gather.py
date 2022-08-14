# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Any

from brt.runtime import log
from brt.router.base import RouterBase, register_router
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
from brt.runtime.proto_tensor import ProtoTensor

__all__ = [
    "GatherRouter",
]

logger = log.get_logger(__file__)

@register_router("gather")
class GatherRouter(RouterBase):
    def __init__(
        self,
        fabric_type: str = "combine",
        fabric_kwargs: Dict[str, Any] = None,
        capaturing=True,
    ):
        """gather router

        Args:
            fabric_type (str, optional): fabric type. Defaults to "combine".
            supported keyword args for fabric:
                reduction (str, optional): reduction method. Defaults to "add".
                sparse (bool, optional): whether restore with zero paddings. Defaults to False.
                auto_padding (bool, optional): whether to pad the tensor to the max shape. Defaults to False.
        """
        super().__init__(capaturing=capaturing)
        self.fabric_type = fabric_type
        self.fabric_kwargs = {}

        if fabric_type == "combine":
            bult_in_fabric_kwargs = {
                "flow_num": 1,
                "reduction": "add",
                "sparse": False,
                "granularity_padding": False,
            }
        self.fabric_kwargs.update(bult_in_fabric_kwargs)

        if fabric_kwargs is not None:
            self.fabric_kwargs.update(fabric_kwargs)

        self.fabric = make_fabric(fabric_type, self.fabric_kwargs)

    def forward(self, in_flows):
        out_flow = self.fabric(in_flows)
        return out_flow


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
    ):
        super().__init__()
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
