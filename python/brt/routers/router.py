# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from brt._C.router import generate_dst_indices
from brt.common import log
from brt.frontend import router

from .fabrics import make_fabric
from .proto_tensor import ProtoTensor
from .protocol import make_protocol
from .symbolic import symbolic_gather_route, symbolic_scatter_route

__all__ = [
    "BaseRouter",
    "ScatterRouter",
    "GatherRouter",
]

logger = log.get_logger(__file__)


@router
class BaseRouter(nn.Module):
    def __init__(self, path_num: int):
        """_summary_

        Args:
            path_num (int): number of paths for routing source or destinations
            gran_dim (_type_, optional): routing granularity. should be a int or a list of int.
        """
        super().__init__()
        self.path_num = path_num

    def route(self, *inputs):
        raise NotImplementedError

    def symbolic_route(self, *inputs):
        raise NotImplementedError


@router
class ScatterRouter(BaseRouter):
    def __init__(
        self,
        path_num: int,
        protocol_type: str = "topk",
        fabric_type: str = "dispatch",
        **kwargs,
    ):
        """base scatter router

        Args:
            path_num (int): number of paths for routing destinations

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
        super().__init__(path_num=path_num)
        self.protocol_type = protocol_type
        self.fabric_type = fabric_type
        self.protocol = make_protocol(protocol_type, path_num=path_num, **kwargs)
        self.fabric = make_fabric(fabric_type, path_num=path_num, **kwargs)

    def route(
        self, in_flow: Union[torch.Tensor, ProtoTensor], score: torch.Tensor
    ) -> List[ProtoTensor]:

        hot_mask = self.protocol(score)

        out_flows = self.fabric(in_flow, hot_mask, score)

        return out_flows

    def symbolic_route(
        self, inputs: torch.Tensor, score: torch.Tensor
    ) -> List[torch.Tensor]:
        return symbolic_scatter_route(inputs, self.path_num)


@router
class GatherRouter(BaseRouter):
    def __init__(
        self,
        path_num: int,
        fabric_type: str = "combine",
        **kwargs,
    ):
        """gather router

        Args:
            path_num (int): number of paths for routing sources
            reduction (str, optional): reduction method. Defaults to "add".
            sparse (bool, optional): whether restore with zero paddings. Defaults to True.
        """
        super().__init__(path_num=path_num)
        self.fabric_type = fabric_type
        self.fabric = make_fabric(fabric_type, path_num=path_num, **kwargs)

    def route(self, in_flows: List[ProtoTensor]) -> ProtoTensor:
        out_flow = self.fabric(in_flows)
        return out_flow

    def symbolic_route(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return symbolic_gather_route(inputs, self.path_num)
