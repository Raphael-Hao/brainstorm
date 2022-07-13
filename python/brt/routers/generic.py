# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union

import torch
from brt.common import log
from brt.routers.fabrics import make_fabric
from brt.routers.proto_tensor import ProtoTensor
from brt.routers.protocols import make_protocol
from brt.routers.router import RouterBase, register_router

__all__ = [
    "ScatterRouter",
    "GatherRouter",
]

logger = log.get_logger(__file__)


@register_router("scatter")
class ScatterRouter(RouterBase):
    def __init__(
        self,
        protocol_type: str = "topk",
        fabric_type: str = "dispatch",
        **kwargs,
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
                        2d: route along the first 2 dimensions, selecting data from a tensor with shape (batch_size, dst_num, ...)
                    transform (bool, optional): whether to transform the route result to the original shape. Defaults to False.

        """
        super().__init__()
        self.protocol_type = protocol_type
        self.fabric_type = fabric_type
        self.protocol = make_protocol(protocol_type, **kwargs)
        self.fabric = make_fabric(fabric_type, **kwargs)

    def forward(
        self, in_flow: Union[torch.Tensor, ProtoTensor], score: torch.Tensor
    ) -> List[ProtoTensor]:

        decisions = self.protocol(score)

        out_flows = self.fabric(in_flow, decisions, score)

        return out_flows

    def update_protocol(self, **kwargs):
        self.protocol.update(**kwargs)


@register_router("gather")
class GatherRouter(RouterBase):
    def __init__(
        self,
        fabric_type: str = "combine",
        **kwargs,
    ):
        """gather router

        Args:
            fabric_type (str, optional): fabric type. Defaults to "combine".
            supported keyword args for fabric:
                reduction (str, optional): reduction method. Defaults to "add".
                sparse (bool, optional): whether restore with zero paddings. Defaults to True.
        """
        super().__init__()
        self.fabric_type = fabric_type
        self.fabric = make_fabric(fabric_type, **kwargs)

    def forward(self, in_flows: List[ProtoTensor]) -> ProtoTensor:
        out_flow = self.fabric(in_flows)
        return out_flow


@register_router("loop")
class LoopRouter(RouterBase):
    def __init__(self, netlet, protocol_type: str = "topk", **kwargs):
        super().__init__()
        self.protocol_type = protocol_type
        self.protocol = make_protocol(self.protocol_type, **kwargs)
        self.dispatch_fabric = make_fabric("dispatch", **kwargs)
        self.combine_fabric = make_fabric("combine", **kwargs)
        self.netlet = netlet

    def forward(self, in_flow: ProtoTensor) -> ProtoTensor:
        target = in_flow
        pending_combine_flows = []
        while target.numel() > 0:
            target, score = self.netlet(target)
            decisions = self.protocol(score)
            dispatched_flows = self.dispatch_fabric(target, decisions)
            target = dispatched_flows[0]
            pending_combine_flows.append(dispatched_flows[1])
        out_flows = self.combine_fabric(pending_combine_flows)
        return out_flows

    def update_protocol(self, **kwargs):
        self.protocol.update(**kwargs)
