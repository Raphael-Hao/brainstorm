# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import time
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from brt.common import log
from brt.primitive import router
from tutel.impls.fast_dispatch import TutelMoeFastDispatcher, extract_critical

from .base import BaseRouter
from .dispatcher import DefaultDispatcher, FusedDispatcher
from .symbolic import symbolic_scatter_route

__all__ = [
    "ScatterRouter",
    "RandomScatterRouter",
    "FusedRandomScatterRouter",
]

logger = log.get_logger(__file__)


@router
class ScatterRouter(BaseRouter):
    def __init__(
        self,
        route_num,
        route_func,
        route_method="topk",
        route_method_args=1,
        dispatcher_cls=DefaultDispatcher,
        transform=True,
        **kwargs,
    ):
        """base scatter router

        Args:
            route_num (int): routing number
            gran_dim (int, optional): routing grainularity. Defaults to None.
            route_method = "topk", "threshold"
                topk: select the topk of gate results as the route destinations
                threshold: select the gate results that are larger than threshold as the route destinations
            transform (bool, optional): whether to transform the routing results. Defaults to True.
        """
        super().__init__(route_num=route_num)
        self.route_func = route_func
        self.route_method = route_method
        self.route_method_args = route_method_args
        self.transform = transform
        self.dispatcher = dispatcher_cls(self.route_num, transform=transform, **kwargs)

    def route(
        self, inputs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """should be implemented by all scatter routers

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: reverse indices for each routing dst
            int: Loads
        """
        loads = inputs.size(0)
        gates = self.route_func(inputs)
        if self.route_method == "topk":
            route_indices = torch.topk(gates, self.route_method_args, dim=1).indices
            route_indices = torch.zeros(
                inputs.size(0), self.route_num, dtype=torch.int64, device=inputs.device
            ).scatter_(1, route_indices, 1)
        elif self.route_method == "threshold":
            route_indices = (gates > self.route_method_args).long().to(inputs.device)
        route_results, route_tags = self.dispatcher.dispatch(
            inputs, route_indices, gates
        )
        return route_results, route_tags, loads

    def symbolic_route(
        self, inputs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        return symbolic_scatter_route(inputs, self.route_num)


@router
class SparseScatterRouter(BaseRouter):
    def __init__(
        self,
        route_num,
        route_func,
        route_method="topk",
        route_method_args=1,
        transform=True,
        dispatcher_cls=DefaultDispatcher,
    ):
        """base scatter router

        Args:
            route_num (int): routing number
            gran_dim (int, optional): routing grainularity. Defaults to None.
            route_method = "topk", "threshold"
                topk: select the topk of gate results as the route destinations
                threshold: select the gate results that are larger than threshold as the route destinations
            transform (bool, optional): whether to transform the routing results. Defaults to True.
        """
        super().__init__(route_num=route_num)
        self.route_func = route_func
        self.route_method = route_method
        self.route_method_args = route_method_args
        self.transform = transform
        self.dispatcher = dispatcher_cls(self.route_num, transform=transform)

    def route(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """should be implemented by all scatter routers

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: reverse indices for each routing dst
            Union[int, torch.Size]]: indicate the reverse shape info
        """
        loads = inputs.size(0)
        gates = self.route_func(inputs)
        if self.route_method == "topk":
            route_indices = torch.topk(gates, self.route_method_args, dim=1).indices
            route_indices = torch.zeros(inputs.size(0), self.route_num).scatter_(
                1, route_indices, 1
            )
        elif self.route_method == "threshold":
            route_indices = (gates > self.route_method_args).long()
        route_results, reverse_indices = self.dispatcher.dispatch(
            inputs, route_indices, gates
        )
        return route_results, reverse_indices, loads

    def symbolic_route(
        self, inputs: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        return symbolic_scatter_route(inputs, self.route_num)


@router
class RandomScatterRouter(ScatterRouter):
    def __init__(
        self,
        route_num: int,
        dispatcher_cls=DefaultDispatcher,
    ):
        """random scatter router

        Args:
            route_num (int): routing number
        """

        def route_func(inputs):
            gates = torch.randn((inputs.size(0), route_num))
            return gates

        super().__init__(
            route_num=route_num,
            route_func=route_func,
            route_method="topk",
            route_method_args=1,
            dispatcher_cls=dispatcher_cls,
            transform=False,
        )


@router
class FusedRandomScatterRouter(ScatterRouter):
    def __init__(
        self,
        route_num: int,
        supported_capacities: List[int],
    ):
        """random scatter router

        Args:
            route_num (int): routing number
        """
        super().__init__(route_num=route_num)
        self.supported_capacities = supported_capacities
        self.dispatcher = FusedDispatcher(self.route_num, supported_capacities)

    def route(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """routing logic of random router

        Args:
            inputs (_type_): inputs for routing

        Returns:
            Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: reverse indices for each routing dst
            Union[int, torch.Size]]: indicate the reverse shape info
        """

        # calculate the scatter indices
        route_dsts = torch.randint(
            0, self.route_num, (inputs.size(0),), device=inputs.device
        )

        # dispatch according to the indices
        results, reverse_indices, capacities = self.dispatcher.dispatch(
            inputs=inputs, route_dsts=route_dsts
        )
        return results, reverse_indices, capacities
