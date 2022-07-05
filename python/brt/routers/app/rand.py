# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from functools import partial
from typing import List, Tuple, Union

import torch
from brt.frontend import nn, router
from brt.routers import GatherRouter, ScatterRouter, reset_proto_tensor_cls
from brt.routers.fabric import (
    HomoFusedGatherRouter,
    HomoFusedScatterRouter,
    make_homo_proto_tensor_cls,
)

__all__ = [
    "init_rand_router",
    "RandScatterRouter",
    "RandGatherRouter",
    "init_rand_homo_fused_router",
    "RandHomoFusedScatterRouter",
    "RandHomoFusedGatherRouter",
]


class RandomGate(nn.Module):
    def __init__(self, dst_num: int) -> None:
        super().__init__()
        self.dst_num = dst_num

    def forward(self, x):
        return torch.randn((x.size(0), self.dst_num), device=x.device)


def init_rand_router():
    reset_proto_tensor_cls()


def init_rand_homo_fused_router():
    make_homo_proto_tensor_cls()


class RandScatterRouter(nn.Module):
    def __init__(
        self,
        dst_num: int,
    ):
        """random scatter router

        Args:
            dst_num (int): routing number
        """

        super().__init__()
        self.rand_gate = RandomGate(dst_num=dst_num)
        self.scatter_router = ScatterRouter(dst_num, gate_method="topk", k=1)

    def forward(self, inputs):
        score = self.score_func(inputs)
        route_results = self.scatter_router(inputs, score)
        return route_results


@router
class RandGatherRouter(GatherRouter):
    def __init__(
        self,
        dst_num: int,
    ):
        """random scatter router

        Args:
            dst_num (int): routing number
        """

        super().__init__(path_num=dst_num)


class RandHomoFusedScatterRouter(nn.Module):
    def __init__(self, dst_num: int, supported_capacities):
        """random scatter router

        Args:
            dst_num (int): routing number
        """

        super().__init__(
            dst_num=dst_num,
            route_func=partial(_rand_route_func, dst_num=dst_num),
            route_method="topk",
            transform=False,
            supported_capacities=supported_capacities,
            k=1,
        )


@router
class RandHomoFusedGatherRouter(HomoFusedGatherRouter):
    def __init__(
        self,
        dst_num: int,
    ):
        """random scatter router

        Args:
            dst_num (int): routing number
        """

        super().__init__(
            dst_num=dst_num,
            transform=False,
        )
