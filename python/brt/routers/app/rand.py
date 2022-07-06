# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from functools import partial
from typing import List, Tuple, Union

import torch
from brt.frontend import nn, router
from brt.routers import GatherRouter, ScatterRouter

__all__ = [
    "RandScatterRouter",
]


class RandomGate(nn.Module):
    def __init__(self, path_num: int) -> None:
        super().__init__()
        self.path_num = path_num

    def forward(self, x):
        return torch.randn((x.size(0), self.path_num), device=x.device)


class RandScatterRouter(nn.Module):
    def __init__(self, path_num: int, fabric_type: str = "dispatch", **kwargs):
        """random scatter router

        Args:
            path_num (int): number of paths for routing destinations
            fabric_type (str, optional): fabric type. Defaults to "dispatch".
                dispatch: dispatch the results to each path, independently.
                homo_fused_dispatch: dispatch the results to all paths in the form of a fused tensor
                    supported_capacities: (List[int]): required for homo_fused_dispatch
        """

        super().__init__()
        self.gate = RandomGate(path_num=path_num)
        self.scatter_router = ScatterRouter(
            path_num, protocol_type="topk", fabric_type=fabric_type, k=1, **kwargs
        )

    def forward(self, inputs):
        score = self.gate(inputs)
        route_results = self.scatter_router(inputs, score)
        return route_results
