# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, Tuple, List
import torch
import torch.nn.functional as F
from tutel.impls.fast_dispatch import TutelMoeFastDispatcher, extract_critical


from .base import Router

__all__ = ["ScatterRouter", "RandomScatterRouter", "TopKScatterRouter"]


class ScatterRouter(Router):
    def __init__(self, route_num: int, grain_dim: int = None, dtype=None):
        """base scatter router

        Args:
            route_num (int): routing number
            grain_dim (int, optional): routing grainularity. Defaults to None.
        """
        super().__init__(route_num=route_num, grain_dim=grain_dim, dtype=dtype)

    def route(
        self, *inputs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Union[int, torch.Size]]:
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


class RandomScatterRouter(ScatterRouter):
    def __init__(self, route_num: int, grain_dim: int = None, dtype=None):
        """random scatter router

        Args:
            route_num (int): routing number
        """
        super().__init__(route_num=route_num, grain_dim=grain_dim, dtype=dtype)

    def route(
        self, inputs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Union[int, torch.Size]]:
        """routing logic of random router

        Args:
            inputs (_type_): inputs for routing

        Returns:
            Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: reverse indices for each routing dst
            Union[int, torch.Size]]: indicate the reverse shape info
        """
        inputs_size, inputs_shape, _ = self.inspect_inputs(inputs)
        route_indices = torch.randint(0, self.route_num, (inputs_size,))
        route_results = [None] * self.route_num
        reverse_indices = [None] * self.route_num
        for i in range(self.route_num):
            indices = torch.nonzero(route_indices == i)[0]
            if len(indices) > 0:
                tmp_results = [inputs[j] for j in indices]
                # TODO: current only support tensors with same shape
                route_results[i] = torch.stack(tmp_results)
                reverse_indices[i] = indices
        return route_results, reverse_indices, inputs_shape


class TopKScatterRouter(ScatterRouter):
    def __init__(
        self,
        route_num: int,
        grain_dim: int,
        k=2,
        top_fn=None,
        dispatcher=None,
        dtype=None,
    ):
        """top-K scatter router

        Args:
            route_num (int): routing number
            grain_dim (int): routing grainularity
            k (int, optional): K. Defaults to 2.
            top_fn (_type_, optional): function for top-k mapping. None defaults to nn.linear.
        """
        super().__init__(route_num=route_num, grain_dim=grain_dim, dtype=dtype)
        self.grain_dim = grain_dim
        assert self.grain_dim > 0, f"grain dim {self.grain_dim} is not valid"
        self.k = min(k, route_num)
        assert self.k > 0, f"Top-K value {self.k} is not valid"
        self.using_tutel = False
        if dispatcher is not None:
            self.dispatcher = dispatcher
        else:
            self.dispatcher = TutelMoeFastDispatcher(
                num_global_experts=self.route_num,
                capacity=self.k,
                model_dim=self.grain_dim,
                dispatch_dtype=self.dtype,
            )
        if isinstance(self.dispatcher, TutelMoeFastDispatcher):
            print(
                "The Top-K scatter router is using tutel dispatcher, the dispatcher should be shared with the corresponding gather router"
            )
            self.using_tutel = True
        if top_fn == None:
            self.top_fn = torch.nn.Linear(self.grain_dim, self.route_num, bias=False)
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
        print(f"indices: {critical_data[1]}")
        print(f"locations: {critical_data[2]}")
        print(f"gates: {critical_data[3]}")
        print(f"capacity: {critical_data[4]}")
        # * using default update method => is_postscore = True
        self.dispatcher.update(*critical_data[1:])
        route_results = self.dispatcher.encode(inputs)
        return route_results, l_loss
