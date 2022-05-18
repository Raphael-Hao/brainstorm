# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
from brt.common import log

from .base import Dispatcher

logger = log.get_logger(__file__)


class FusedDispatcher(Dispatcher):
    """
    Default dispatcher implement by Brainstorm
    The default dispatcher will dispatch the inputs to the routers just according to the ground-truth routing indices.
    """

    def __init__(self, route_num, supported_capacities):
        super().__init__(route_num)
        self.supported_capacities = [0]
        if supported_capacities is not None:
            self.supported_capacities.extend(supported_capacities)

    def capacity(self, branch_accept):
        """
        Return the capacity of each branch
        """
        for i, supported_capcity in enumerate(self.supported_capacities):
            if branch_accept <= supported_capcity:
                return supported_capcity
        logger.error(f"No capacity found for this branch accept: {branch_accept}")

    def dispatch(self, inputs: torch.Tensor, route_dsts: torch.Tensor):
        """
        Dispatch the inputs into the the list of torch.Tensor with indices
        """
        # dsts[0, 1, 2, 1, 2, 3] -> route_capacities[1, 2, 2, 1]
        logger.debug("input route_dsts:", route_dsts)
        route_capacities = torch.bincount(route_dsts)
        logger.debug("route_capacities:", route_capacities)
        total_capacity = 0
        # route_capacities[1, 2, 2, 1] -> route_capacities[2, 2, 2, 2]
        for i in range(len(route_capacities)):
            capacity = self.capacity(route_capacities[i])
            route_capacities[i] = capacity
            total_capacity += capacity
        logger.debug(
            f"padded route capacities: {route_capacities} , total capacity: {total_capacity}",
        )
        # route_shape = torch.Size([total_capacity] + list(inputs.shape[1:]))
        route_start = torch.zeros_like(route_capacities)
        start_idx = 0
        # route_start[0, 2, 4, 6]
        for i in range(len(route_capacities)):
            route_start[i] = start_idx
            start_idx += route_capacities[i]
        # print("route_start:", route_start)
        # dsts[0, 1, 2, 1, 2, 3] -> dsts[0, 2, 4, 3, 5, 6]
        for i in range(route_dsts.numel()):
            start_idx = route_start[route_dsts[i]].item()
            route_start[route_dsts[i]] += 1
            route_dsts[i] = start_idx
        logger.debug("reverse indices:", route_dsts)

        repeat_size = inputs.numel() // inputs.size(0)
        route_indices = route_dsts.view(-1, 1).repeat(1, repeat_size).view_as(inputs)
        route_results = torch.zeros(total_capacity, *inputs.shape[1:])
        route_results = torch.scatter(route_results, 0, route_indices, inputs)
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
