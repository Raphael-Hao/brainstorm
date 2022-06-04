# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union

import numpy as np
import torch
from brt.common import log
from brt.primitive import router

from .base import BaseRouter
from .flow_tensor import FlowTensor, deinit_flow_tensor, init_flow_tensor
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
        dst_num,
        route_func,
        route_method: str = "topk",
        residual_dst: int = -1,
        transform=True,
        sparse=True,
        **kwargs,
    ):
        """base scatter router

        Args:
            dst_num (int): routing number
            route_method = "topk", "threshold"
                topk: select the topk of gate results as the route destinations
                threshold: select the gate results that are larger than threshold as the route destinations
            dispatcher_cls (type): dispatcher class
        """
        super().__init__(dst_num=dst_num)
        self.route_func = route_func
        self.route_method = route_method.lower()
        self.sparse = sparse
        self.residual_dst = residual_dst
        self.transform = transform
        if self.route_method == "topk":
            self.k = kwargs.get("k")
            if self.k == None:
                self.k = 1
                logger.warning(
                    "k is not specified for Top-K route method, use default k=1"
                )
        elif self.route_method == "threshold":
            self.threshold = kwargs.get("threshold")
            if self.threshold == None:
                self.threshold = 0.0
                logger.warning(
                    "threshold is not specified for Threshold route method, use default threshold=0.0"
                )
        else:
            raise ValueError(f"route_method {route_method} is not supported")

    def route(self, inputs: Union[torch.Tensor, FlowTensor]) -> List[FlowTensor]:
        """should be implemented by all scatter routers

        Returns:
            List[torch.Tensor]: routing results for each routing dst
            List[torch.Tensor]: routing tags for each routing dst
            int: Loads
        """
        inputs = self.tag_inputs(inputs)
        route_indices, gates = self.gen_indices_and_gates(inputs)
        route_results = self.dispatch(inputs, route_indices, gates)
        return route_results

    def tag_inputs(self, inputs: Union[torch.Tensor, FlowTensor]) -> FlowTensor:
        """tag inputs with routing tags"""
        if isinstance(inputs, FlowTensor):
            if inputs.size(0) != inputs.tag.numel():
                # route granularity changed, we will re-tag the inputs
                new_tag = torch.arange(
                    0, inputs.size(0), dtype=torch.int64, device=inputs.device
                ).view(-1, 1)
                inputs.init_flow(new_tag, load=new_tag.numel())
            return inputs
        elif isinstance(inputs, torch.Tensor):
            tag = torch.arange(
                0, inputs.size(0), dtype=torch.int64, device=inputs.device
            ).view(-1, 1)
            inputs = init_flow_tensor(inputs, tag, load=tag.numel())
            return inputs
        else:
            raise ValueError(f"unsupported input type {type(inputs)}")

    def gen_indices_and_gates(
        self, inputs_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """generate gates with indices

        Args:
            inputs (torch.Tensor): input tensor
        """
        gates = self.route_func(inputs_data)
        gates.__class__ = torch.Tensor
        assert gates.size(0) == inputs_data.size(0) and gates.size(1) == self.dst_num
        if self.route_method == "topk":
            route_indices = torch.topk(gates, self.k, dim=1).indices
            route_indices = torch.zeros(
                gates.size(0),
                self.dst_num,
                dtype=torch.int64,
                device=inputs_data.device,
            ).scatter_(1, route_indices, 1)
        elif self.route_method == "threshold":
            route_indices = (gates > self.threshold).long().to(inputs_data.device)
            if self.residual_dst >= 0:
                residual_indices = (
                    (route_indices.sum(dim=1, keepdim=True) == 0)
                    .long()
                    .to(inputs_data.device)
                )
                residual_index = torch.full(
                    (residual_indices.shape),
                    self.residual_dst,
                    dtype=torch.int64,
                    device=inputs_data.device,
                )
                route_indices = torch.scatter_add(
                    route_indices, 1, residual_index, residual_indices
                )
        return route_indices, gates

    def dispatch(
        self,
        inputs: FlowTensor,
        route_indices: torch.Tensor,
        gates: torch.Tensor,
    ) -> List[FlowTensor]:
        """
        Dispatch the inputs into the the list of torch.Tensor with indices
        """
        route_data = inputs.data
        route_tag = inputs.tag
        load = inputs.load
        route_shape = list(route_data.shape[1:])
        route_size = np.prod(route_shape)

        results = [
            init_flow_tensor(
                data=torch.zeros(
                    0, *route_shape, dtype=route_data.dtype, device=route_data.device
                ),
                tag=torch.zeros(0, 1, dtype=torch.int64, device=route_data.device),
                load=load,
            )
            for _ in range(self.dst_num)
        ]
        for i in range(self.dst_num):
            tag_indices = torch.nonzero(route_indices[:, i].view(-1)).to(
                route_data.device
            )
            if tag_indices.numel() > 0:
                tag = torch.gather(route_tag, 0, tag_indices)
                data_indices = tag_indices.repeat(1, route_size).view(-1, *route_shape)
                if self.transform:
                    gate = gates[:, i].reshape(
                        (route_data.size(0),) + (1,) * len(route_shape)
                    )
                    data = torch.gather(route_data * gate, 0, data_indices)
                else:
                    data = torch.gather(route_data, 0, data_indices)
                results[i] = init_flow_tensor(data, tag, load)
            if self.sparse:
                results[i].load = tag_indices.numel()

        return results

    def symbolic_route(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return symbolic_scatter_route(inputs, self.dst_num)


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
