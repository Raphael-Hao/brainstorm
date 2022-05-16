# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from brt.common import log
from brt.primitive import router
from tutel.impls.fast_dispatch import TutelMoeFastDispatcher, extract_critical

from .base import BaseRouter
from .dispatcher import DefaultDispatcher

__all__ = ["ScatterRouter", "RandomScatterRouter", "TopKScatterRouter"]

logger = log.get_logger(__file__)


@router
class ScatterRouter(BaseRouter):
    def __init__(self, route_num: int, gran_dim: int = None, dtype=None):
        """base scatter router

        Args:
            route_num (int): routing number
            gran_dim (int, optional): routing grainularity. Defaults to None.
        """
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)

    def route(
        self, inputs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """should be implemented by all scatter routers

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: reverse indices for each routing dst
            Union[int, torch.Size]]: indicate the reverse shape info
        """
        raise NotImplementedError

    def record(self):
        self.active_counter += 1

    def register_router(self):
        pass


@router
class RandomScatterRouter(ScatterRouter):
    def __init__(
        self, route_num: int, gran_dim: int = None, dispatcher=None, dtype=None
    ):
        """random scatter router

        Args:
            route_num (int): routing number
        """
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
        self.dispatcher = (
            DefaultDispatcher(self.route_num) if dispatcher is None else dispatcher
        )

    def route(
        self, inputs
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
        inputs_size, inputs_shape, _ = self.inspect_inputs(inputs)
        route_indices = torch.randint(0, self.route_num, (inputs_size,))

        # dispatch according to the indices
        route_results, reverse_indices = self.dispatcher.dispatch(
            inputs=inputs, route_indices=route_indices
        )
        return route_results, reverse_indices, inputs_shape

    def symbolic_route(self, inputs):
        return torch.ops.brt.symbolic_scatter_route(inputs, 0, self.route_num)


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
        dispatcher=None,
        dtype=None,
    ):
        """multiplex scatter router

        Args:
            route_num (int): routing number
            gran_dim (int): routing grainularity
            multiplex_fn (_type_, optional): function for multiplex mapping. None defaults to nn.linear.
        """
        super().__init__(route_num=route_num, gran_dim=gran_dim, dtype=dtype)
        if dispatcher is not None:
            self.dispatcher = dispatcher
        else:
            self.dispatcher = TutelMoeFastDispatcher(
                num_global_experts=self.route_num,
                capacity=self.route_num,
                model_dim=self.gran_dim,
                dispatch_dtype=self.dtype,
            )
        if isinstance(self.dispatcher, TutelMoeFastDispatcher):
            print(
                "The Multiplex scatter router is using tutel dispatcher, the dispatcher should be shared with the corresponding gather router"
            )
        if multiplex_fn == None:
            self.multiplex_fn = torch.nn.Linear(
                self.gran_dim, self.route_num, bias=False
            )
        else:
            self.multiplex_fn = multiplex_fn

    def route(self, inputs):
        """routing logic of multiplex router

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: reverse indices for each routing dst
            Union[int, torch.Size]]: indicate the reverse shape info
            torch.Tensor: loss of top-k for load balance
        """
        pass
