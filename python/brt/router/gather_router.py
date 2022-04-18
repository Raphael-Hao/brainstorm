# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union
import torch


from .base import Router

__all__ = ["GatherRouter", "RandomGatherRouter", "TopKGatherRouter"]


class GatherRouter(Router):
    def __init__(self, route_num: int, grain_dim: int = None, dtype=None):
        super().__init__(route_num=route_num, grain_dim=grain_dim, dtype=dtype)

    def route(
        self,
        reverse_indices: List[torch.Tensor],
        reverse_shape: Union[int, torch.Size],
        *inputs,
    ):
        raise NotImplementedError

    def record(self):
        self.active_counter += 1

    def register_router(self):
        pass


class RandomGatherRouter(GatherRouter):
    def __init__(self, route_num: int, grain_dim: int = None, dtype=None):
        super().__init__(route_num=route_num, grain_dim=grain_dim, dtype=dtype)

    def route(
        self,
        reverse_indices: List[torch.Tensor],
        reverse_shape: Union[int, torch.Size],
        *inputs,
    ):
        assert (
            len(inputs) == self._route_num and len(reverse_indices) == self._route_num
        )
        if isinstance(reverse_shape, int):
            route_size = reverse_shape
        elif isinstance(reverse_shape, torch.Size):
            route_size = reverse_shape[0]
        else:
            raise ValueError("origin_shape must be a int or torch.Size")
        route_results = [[] for _ in range(route_size)]
        for i in range(self._route_num):
            if reverse_indices[i] is not None:
                for j in range(len(reverse_indices[i])):
                    route_results[reverse_indices[i][j]] = inputs[i][j]
        if isinstance(reverse_shape, int):
            return route_results
        else:
            route_results = torch.stack(route_results).view(reverse_shape)
            return route_results


class TopKGatherRouter(GatherRouter):
    def __init__(self, route_num: int, grain_dim: int, k=2, dtype=None):
        super().__init__(route_num=route_num, grain_dim=grain_dim, dtype=dtype)
        self.k = k

    def route(
        self,
        reverse_indices: List[torch.Tensor],
        reverse_shape: Union[int, torch.Size],
        *inputs: List[torch.Tensor],
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
            size=(route_size, self.grain_dim),
            dtype=valid_input.dtype,
            device=valid_input.device,
        )

        if isinstance(reverse_shape, torch.Size):
            return route_results
        else:
            route_results = torch.chunk(route_results, route_size)
            return route_results
