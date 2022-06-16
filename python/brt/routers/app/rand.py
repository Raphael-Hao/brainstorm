# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from functools import partial
from typing import List, Tuple, Union

import torch
from brt.primitive import router
from brt.routers import reset_proto_tensor_cls
from brt.routers.base import GatherRouter, ScatterRouter
from brt.routers.inference import (
    HomoFusedGatherRouter,
    HomoFusedScatterRouter,
    make_homo_proto_tensor_cls,
)

__all__ = ["RandScatterRouter", "RandHomoFusedScatterRouter"]


def rand_route_func(inputs_data, dst_num):
    gates = torch.randn((inputs_data.size(0), dst_num), device=inputs_data.device)
    return gates


def init_rand_router():
    reset_proto_tensor_cls()
    
def init_rand_homo_fused_router():
    make_homo_proto_tensor_cls()

@router
class RandScatterRouter(ScatterRouter):
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
            route_func=partial(rand_route_func, dst_num=dst_num),
            route_method="topk",
            transform=False,
            k=1,
        )


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

        super().__init__(
            dst_num=dst_num,
            transform=False,
        )


@router
class RandHomoFusedScatterRouter(HomoFusedScatterRouter):
    def __init__(self, dst_num: int, supported_capacities):
        """random scatter router

        Args:
            dst_num (int): routing number
        """

        super().__init__(
            dst_num=dst_num,
            route_func=partial(rand_route_func, dst_num=dst_num),
            route_method="topk",
            transform=False,
            supported_capacities=supported_capacities,
            k=4,
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
