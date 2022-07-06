# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from functools import partial
from typing import List, Tuple, Union

import torch
from brt.frontend import nn, router
from brt.routers import GatherRouter, ScatterRouter

__all__ = [
    "RandScatterRouter",
    "RandGatherRouter",
    "RandHomoFusedScatterRouter",
    "RandHomoFusedGatherRouter",
]


class RandomGate(nn.Module):
    def __init__(self, path_num: int) -> None:
        super().__init__()
        self.path_num = path_num

    def forward(self, x):
        return torch.randn((x.size(0), self.path_num), device=x.device)


class RandScatterRouter(nn.Module):
    def __init__(
        self,
        path_num: int,
    ):
        """random scatter router

        Args:
            path_num (int): routing number
        """

        super().__init__()
        self.score_func = RandomGate(path_num=path_num)
        self.scatter_router = ScatterRouter(
            path_num, protocol_type="topk", fabric_type="dispatch", k=1
        )

    def forward(self, inputs):
        score = self.score_func(inputs)
        route_results = self.scatter_router(inputs, score)
        return route_results


@router
class RandGatherRouter(GatherRouter):
    def __init__(
        self,
        path_num: int,
    ):
        """random scatter router

        Args:
            path_num (int): routing number
        """

        super().__init__(path_num=path_num)


class RandHomoFusedScatterRouter(nn.Module):
    def __init__(self, path_num: int, supported_capacities):
        """random scatter router

        Args:
            path_num (int): routing number
        """

        super().__init__()
        self.score_func = RandomGate(path_num=path_num)
        self.scatter_router = ScatterRouter(
            path_num=path_num,
            protocol_type="topk",
            fabric_type="homo_dispatch",
            transform=True,
            supported_capacities=supported_capacities,
        )


@router
class RandHomoFusedGatherRouter(GatherRouter):
    def __init__(
        self,
        path_num: int,
    ):
        """random scatter router

        Args:
            path_num (int): routing number
        """

        super().__init__(
            path_num=path_num,
            transform=False,
        )
