# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch
from brt.primitive import router

from .base import BaseRouter
from .dispatcher import DefaultDispatcher

__all__ = ["GatherRouter", "RandomGatherRouter", "TopKGatherRouter"]


@router
class GatherRouter(BaseRouter):
    def __init__(self, route_num: int, gran_dim: int = None, dtype=None):
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)

    def route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def record(self):
        self.active_counter += 1

    def register_router(self):
        pass


@router
class RandomGatherRouter(GatherRouter):
    def __init__(
        self, route_num: int, gran_dim: int = None, dispatcher=None, dtype=None
    ):
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
        self.dispatcher = (
            DefaultDispatcher(self.route_num) if dispatcher is None else dispatcher
        )

    def route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ) -> torch.Tensor:
        route_results = self.dispatcher.combine(reverse_indices, reverse_shape, inputs,)
        return route_results

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ):
        return torch.ops.brt.symbolic_gather_route(
            inputs, reverse_indices, reverse_shape, 0, self.route_num
        )


@router
class TopKGatherRouter(GatherRouter):
    def __init__(self, route_num: int, gran_dim: int, k=2, dtype=None):
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
        self.k = k

    def route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ):
        if isinstance(reverse_shape, int):
            route_size = reverse_shape
        elif isinstance(reverse_shape, torch.Size):
            route_size = reverse_shape[0]
        else:
            raise ValueError("origin_shape must be a int or torch.Size")

        valid_input = next(_input for _input in inputs if _input is not None)
        assert (
            self.dtype == valid_input.dtype
        ), "Dtype of router and input must be consistent"
        route_results = torch.zeros(
            size=(route_size, self.gran_dim),
            dtype=valid_input.dtype,
            device=valid_input.device,
        )

        if isinstance(reverse_shape, torch.Size):
            return route_results
        else:
            route_results = torch.chunk(route_results, route_size)
            return route_results

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ):
        return torch.ops.brt.symbolic_gather_route(
            inputs, reverse_indices, reverse_shape, self.k, self.route_num
        )


@router
class AggregateGatherRouter(GatherRouter):
    def __init__(self, route_num: int, gran_dim: int, dtype=None):
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)

    def route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ):
        if isinstance(reverse_shape, int):
            route_size = reverse_shape
        elif isinstance(reverse_shape, torch.Size):
            route_size = reverse_shape[0]
        else:
            raise ValueError("origin_shape must be a int or torch.Size")

        valid_input = next(_input for _input in inputs if _input is not None)
        assert (
            self.dtype == valid_input.dtype
        ), "Dtype of router and input must be consistent"
        route_results = torch.zeros(
            size=(route_size, self.gran_dim),
            dtype=valid_input.dtype,
            device=valid_input.device,
        )

        if isinstance(reverse_shape, torch.Size):
            return route_results
        else:
            route_results = torch.chunk(route_results, route_size)
            return route_results

    def symbolic_route(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        reverse_shape: torch.Tensor,
    ):
        return torch.ops.brt.symbolic_gather_route(
            inputs, reverse_indices, reverse_shape, 3, self.route_num
        )
