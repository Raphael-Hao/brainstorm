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
    "TopKScatterRouter",
    "MultiplexScatterRouter",
]

logger = log.get_logger(__file__)


def get_scatter_router_kind(route_method, route_method_args):
    if route_method == "topk":
        return route_method_args
    elif route_method == "threshold":
        return 0


@router
class ScatterRouter(BaseRouter):
    def __init__(
        self,
        route_num,
        gran_dim: Union[int, List[int]],
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
        super().__init__(route_num=route_num, gran_dim=gran_dim)
        self.route_func = route_func
        self.route_method = route_method
        self.route_method_args = route_method_args
        self.transform = transform
        self.router_kind = get_scatter_router_kind(route_method, route_method_args)
        self.dispatcher = dispatcher_cls(
            self.route_num, self.gran_dim, transform=transform
        )

    def route(
        self, inputs: torch.Tensor
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

    def symbolic_route(self, inputs):
        return symbolic_scatter_route(inputs, self.router_kind, self.route_num)


@router
class RandomScatterRouter(ScatterRouter):
    def __init__(
        self,
        route_num: int,
        gran_dim: Union[int, List[int]],
        transform=True,
        DispatcherCLS=DefaultDispatcher,
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
            gran_dim=gran_dim,
            route_func=route_func,
            route_method="topk",
            route_method_args=1,
            transform=True,
        )
        self.dispatcher = DispatcherCLS(self.route_num, self.gran_dim)

    def route(
        self, inputs: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
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
        inputs_shape = torch._shape_as_tensor(inputs)
        route_indices = torch.randint(0, self.route_num, (inputs.size(0),))
        route_indices = F.one_hot(route_indices, self.route_num).long()
        gates = torch.ones((inputs.size(0), self.route_num), dtype=inputs.dtype)

        # dispatch according to the indices
        route_results, reverse_indices = self.dispatcher.dispatch(
            inputs=inputs, route_indices=route_indices
        )
        return route_results, reverse_indices, inputs_shape

    def symbolic_route(self, inputs):
        return symbolic_scatter_route(inputs, 0, self.route_num)


@router
class FusedRandomScatterRouter(ScatterRouter):
    def __init__(
        self,
        route_num: int,
        supported_capacities: List[int],
        gran_dim: int = None,
        dtype=None,
    ):
        """random scatter router

        Args:
            route_num (int): routing number
        """
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
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


@router
class TopKScatterRouter(ScatterRouter):
    def __init__(
        self,
        route_num: int,
        gran_dim: int,
        k=2,
        top_fn=None,
        dispatcher=None,
        dtype=None,
    ):
        """top-K scatter router

        Args:
            route_num (int): routing number
            gran_dim (int): routing grainularity
            k (int, optional): K. Defaults to 2.
            top_fn (_type_, optional): function for top-k mapping. None defaults to nn.linear.
        """
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
        self.k = min(k, self.route_num)
        assert self.k > 0, f"Top-K value {self.k} is not valid"
        self.using_tutel = False
        if dispatcher is not None:
            self.dispatcher = dispatcher
        else:
            self.dispatcher = TutelMoeFastDispatcher(
                num_global_experts=self.route_num,
                capacity=self.k,
                model_dim=self.gran_dim,
                dispatch_dtype=self.dtype,
            )
        if isinstance(self.dispatcher, TutelMoeFastDispatcher):
            logger.debug(
                "The Top-K scatter router is using tutel dispatcher, the dispatcher should be shared with the corresponding gather router"
            )
            self.using_tutel = True
        if top_fn == None:
            self.top_fn = torch.nn.Linear(self.gran_dim, self.route_num, bias=False)
        else:
            self.top_fn = top_fn

    def route(self, inputs):
        """routing logic of top-K router

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: reverse indices for each routing dst
            Union[int, torch.Size]]: indicate the reverse shape info
            torch.Tensor: loss of top-k for load balance
        """

        _inputs_size, inputs_shape, tensor_input = self.inspect_inputs(inputs)
        if tensor_input is not True:
            inputs = torch.stack(inputs)
        route_results, l_loss = self.topk_select_tensor(inputs)

        return route_results, l_loss, inputs_shape

    def topk_select_tensor(self, inputs):
        logits = self.top_fn(inputs)
        gates = F.softmax(logits, dim=1)
        # check why need fp32 gate
        critical_data, l_loss = extract_critical(
            gates, self.k, capacity_factor=1.0, fp32_gate=True
        )
        # print(f"indices: {critical_data[1]}")
        # print(f"locations: {critical_data[2]}")
        # print(f"gates: {critical_data[3]}")
        # print(f"capacity: {critical_data[4]}")
        # * using default update method => is_postscore = True
        before_update_timestamp = time.time()
        self.dispatcher.update(*critical_data[1:])
        logger.debug(f"dispatcher update time: {time.time() - before_update_timestamp}")
        route_results = self.dispatcher.encode(inputs)
        return route_results, l_loss


@router
class MultiplexScatterRouter(ScatterRouter):
    def __init__(
        self,
        route_num: int,
        gran_dim: int,
        multiplex_fn=None,
    ):
        """multiplex scatter router

        Args:
            route_num (int): routing number
            gran_dim (int): routing grainularity
            multiplex_fn (_type_, optional): function for multiplex mapping. None defaults to nn.linear.
        """
        super().__init__(route_num=route_num, gran_dim=gran_dim)
        if multiplex_fn is None and isinstance(gran_dim, int):
            self.multiplex_fn = nn.Sequential(
                nn.Linear(self.gran_dim, self.route_num, bias=False), nn.ReLU()
            )
        elif multiplex_fn is not None:
            self.multiplex_fn = multiplex_fn
        else:
            raise ValueError(
                f"multiplex_fn must be provided if gran_dim is not a 1-D tensor"
            )

    def route(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """routing logic of multiplex router

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: reverse indices for each routing dst
            Union[int, torch.Size]]: indicate the reverse shape info
            torch.Tensor: loss of top-k for load balance
        """
        gates = self.multiplex_fn(inputs)
        route_results = [
            torch.zeros_like(inputs, dtype=self.dtype, device=inputs.device)
            for _ in range(self.route_num)
        ]
        for i in range(self.route_num):
            condition = gates[:, i] > 0
            condition = condition.unsqueeze(1)
            route_results[i] = torch.where(condition, inputs, route_results[i])
        print(f"route_results: {route_results}")
        return route_results

    def symbolic_route(self, inputs):
        route_results, _, _ = symbolic_scatter_route(
            inputs, self.route_num, self.route_num
        )
        return route_results
