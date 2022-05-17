# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

from .base import Dispatcher


class FusedDispatcher(Dispatcher):
    """
    Default dispatcher implement by Brainstorm
    The default dispatcher will dispatch the inputs to the routers just according to the ground-truth routing indices.
    """

    def __init__(self, route_num, supported_capacities):
        super().__init__(route_num)
        self.supported_capacities = supported_capacities

    def capacity(self, branch_accept):
        """
        Return the capacity of each branch
        """
        for i, supported_capcity in enumerate(self.supported_capacities):
            if branch_accept <= supported_capcity:
                return supported_capcity
        return self.supported_capacities[-1]

    def dispatch(self, inputs: torch.Tensor, route_dsts: torch.Tensor):
        """
        Dispatch the inputs into the the list of torch.Tensor with indices
        """
        # dsts[0, 1, 2, 1, 2, 3] -> route_capacities[1, 2, 2, 1]
        # print("input route_dsts:", route_dsts)
        route_capacities = torch.bincount(route_dsts)
        # print("route_capacities:", route_capacities)
        total_capacity = 0
        # route_capacities[1, 2, 2, 1] -> route_capacities[2, 2, 2, 2]
        for i in range(self.route_num):
            capacity = self.capacity(route_capacities[i])
            route_capacities[i] = capacity
            total_capacity += capacity
        # print(
        #     f"padded route capacities: {route_capacities} , total capacity: {total_capacity}",
        # )
        # route_shape = torch.Size([total_capacity] + list(inputs.shape[1:]))
        route_start = torch.zeros_like(route_capacities)
        start_idx = 0
        # route_start[0, 2, 4, 6]
        for i in range(self.route_num):
            route_start[i] = start_idx
            start_idx += route_capacities[i]
        # print("route_start:", route_start)
        # dsts[0, 1, 2, 1, 2, 3] -> dsts[0, 2, 4, 3, 5, 6]
        # indices[0, 1, 2, 3, 4, 5, 6, 7] -> indices[0, 1, 1, 3, 2, 4, 5, 7]
        route_indices = torch.arange(0, total_capacity, dtype=torch.int64)
        for i in range(route_dsts.numel()):
            start_idx = route_start[route_dsts[i]].item()
            route_start[route_dsts[i]] += 1
            route_dsts[i] = start_idx
            route_indices[start_idx] = i
        # print("padded route indices:", route_indices)
        # print("padded reverse indices:", route_dsts)
        # print(route_indices)
        tpd = [0] * inputs.ndim * 2
        tpd[-1] = total_capacity - inputs.shape[0]
        tpd = tuple(tpd)
        padded_inputs = F.pad(inputs, tpd, mode="constant", value=0)
        # print("padd inputs:", padded_inputs)
        repeat_size = inputs.numel() // inputs.size(0)
        route_indices = (
            route_indices.view(-1, 1).repeat(1, repeat_size).view_as(padded_inputs)
        )
        # print(route_indices)
        route_results = torch.gather(padded_inputs, 0, route_indices)
        return route_results, route_dsts, route_capacities

    def combine(
        self, reverse_indices, inputs,
    ):
        """
        Combine the outputs of the routers into the final outputs
        """
        repeat_size = inputs.numel() // inputs.size(0)
        reverse_shape = torch.Size([reverse_indices.size(0)] + list(inputs.shape[1:]))
        reverse_indices = (
            reverse_indices.view(-1, 1).repeat(1, repeat_size).view(reverse_shape)
        )
        route_results = torch.gather(inputs, 0, reverse_indices)
        return route_results
