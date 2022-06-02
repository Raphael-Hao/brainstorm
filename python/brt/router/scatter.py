# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import torch
from brt.common import log
from brt.primitive import router

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
        route_method: str = "topk",
        dispatcher_cls=DefaultDispatcher,
        **kwargs,
    ):
        """base scatter router

        Args:
            route_num (int): routing number
            route_method = "topk", "threshold"
                topk: select the topk of gate results as the route destinations
                threshold: select the gate results that are larger than threshold as the route destinations
            dispatcher_cls (type): dispatcher class
        """
        super().__init__(route_num=route_num)
        self.route_func = route_func
        self.route_method = route_method
        if self.route_method == "topk":
            self.k = kwargs.pop("k", 1)
        elif self.route_method == "threshold":
            self.threshold = kwargs.pop("threshold", 0.0)
            self.residual_dst = kwargs.pop("residual_dst", -1)
        else:
            raise ValueError(f"route_method {route_method} is not supported")
        transform = kwargs.pop("transform", True)
        self.dispatcher = dispatcher_cls(self.route_num, transform=transform, **kwargs)

    def route(
        self, inputs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """should be implemented by all scatter routers

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: routing tags for each routing dst
            int: Loads
        """
        loads = inputs.size(0)
        gates = self.route_func(inputs)
        assert gates.size(0) == loads and gates.size(1) == self.route_num
        if self.route_method == "topk":
            route_indices = torch.topk(gates, self.k, dim=1).indices
            route_indices = torch.zeros(
                inputs.size(0), self.route_num, dtype=torch.int64, device=inputs.device
            ).scatter_(1, route_indices, 1)
        elif self.route_method == "threshold":
            route_indices = (gates > self.threshold).long().to(inputs.device)
            if self.residual_dst >= 0:
                residual_indices = (
                    (route_indices.sum(dim=1, keepdim=True) == 0)
                    .long()
                    .to(inputs.device)
                )
                residual_index = torch.full(
                    (residual_indices.shape),
                    self.residual_dst,
                    dtype=torch.int64,
                    device=inputs.device,
                )
                route_indices = torch.scatter_add(
                    route_indices, 1, residual_index, residual_indices
                )
        route_results, route_tags = self.dispatcher.dispatch(
            inputs, route_indices, gates
        )
        return route_results, route_tags, loads

    def symbolic_route(
        self, inputs: torch.Tensor
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
            dispatcher_cls=dispatcher_cls,
            transform=False,
        )


@router
class SparseScatterRouter(ScatterRouter):
    def __init__(
        self,
        route_num,
        route_func,
        route_method: str = "topk",
        dispatcher_cls=DefaultDispatcher,
        **kwargs,
    ):
        super().__init__(
            route_num=route_num,
            route_func=route_func,
            route_method=route_method,
            dispatcher_cls=dispatcher_cls,
            **kwargs,
        )

    def route(
        self, inputs: torch.Tensor, tags: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """should be implemented by all scatter routers

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: routing tags for each routing dst
            int: Loads
        """
        gates = self.route_func(inputs)
        if self.route_method == "topk":
            route_indices = torch.topk(gates, self.k, dim=1).indices
            route_indices = torch.zeros(
                inputs.size(0), self.route_num, dtype=torch.int64, device=inputs.device
            ).scatter_(1, route_indices, 1)
        elif self.route_method == "threshold":
            route_indices = (gates > self.threshold).long().to(inputs.device)
            if self.residual_dst >= 0:
                residual_indices = (
                    (route_indices.sum(dim=1, keepdim=True) == 0)
                    .long()
                    .to(inputs.device)
                )
                residual_index = torch.full(
                    (residual_indices.shape),
                    self.residual_dst,
                    dtype=torch.int64,
                    device=inputs.device,
                )
                route_indices = torch.scatter_add(
                    route_indices, 1, residual_index, residual_indices
                )
        route_results, route_tags = self.dispatcher.dispatch(
            inputs, route_indices, gates, tags
        )
        return route_results, route_tags

    def symbolic_route(
        self, inputs: torch.Tensor, tags: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        return symbolic_scatter_route(inputs, tags, self.route_num)


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
