# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Tuple, Union

import torch
from brt.primitive import router

from .base import GatherRouter, ScatterRouter


@router
class RandomScatterRouter(ScatterRouter):
    def __init__(
        self,
        dst_num: int,
    ):
        """random scatter router

        Args:
            dst_num (int): routing number
        """

        def route_func(inputs_data):
            gates = torch.randn((inputs_data.size(0), dst_num))
            return gates

        super().__init__(
            dst_num=dst_num,
            route_func=route_func,
            route_method="topk",
            transform=False,
            k=1,
        )


@router
class RandomGatherRouter(GatherRouter):
    def __init__(self, dst_num: int, reduction: str = "add", sparse=True):
        super().__init__(dst_num, reduction, sparse)


@router
class FusedRandomScatterRouter(ScatterRouter):
    def __init__(
        self,
        dst_num: int,
        supported_capacities: List[int],
    ):
        """random scatter router

        Args:
            dst_num (int): routing number
        """
        super().__init__(dst_num=dst_num)
        self.supported_capacities = supported_capacities
        # self.dispatcher = FusedDispatcher(self.dst_num, supported_capacities)

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
            0, self.dst_num, (inputs.size(0),), device=inputs.device
        )

        # dispatch according to the indices
        results, reverse_indices, capacities = self.dispatcher.dispatch(
            inputs=inputs, route_dsts=route_dsts
        )
        return results, reverse_indices, capacities


@router
class FusedRandomGatherRouter(GatherRouter):
    def __init__(self, dst_num: int, gran_dim: int = None, dtype=None):
        super().__init__(dst_num=dst_num, gran_dim=gran_dim, dtype=dtype)

    def route(
        self,
        inputs: torch.Tensor,
        reverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        route_results = self.dispatcher.combine(inputs, reverse_indices)
        return route_results
