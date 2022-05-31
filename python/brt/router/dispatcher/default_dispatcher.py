# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch

from .base import Dispatcher


class DefaultDispatcher(Dispatcher):
    """
    Default dispatcher implement by Brainstorm
    The default dispatcher will dispatch the inputs to the routers just according to the ground-truth routing indices.
    """

    def __init__(self, route_num, gran_dim, transform=True, reduction="add"):
        super().__init__(route_num, gran_dim, transform, reduction)

    def dispatch(self, inputs, route_indices, gates):
        """
        Dispatch the inputs into the the list of torch.Tensor with indices
        """
        route_results = [torch.zeros(0, *self.route_shape)] * self.route_num
        reverse_indices = [torch.zeros(0, 0)] * self.route_num
        for i in range(self.route_num):
            indices = torch.nonzero(route_indices[i])
            if indices.numel() > 0:
                reverse_indices[i] = indices
                indices = indices.repeat(1, self.route_size).view(-1, *self.route_shape)
                if self.transform:
                    gate = gates[:, i].reshape(
                        (inputs.size(0),) + (1,) * len(self.route_shape)
                    )
                    route_results[i] = torch.gather(inputs * gate, 0, indices)
                else:
                    route_results[i] = torch.gather(inputs, 0, indices)
        return route_results, reverse_indices

    def combine(
        self,
        inputs: List[torch.Tensor],
        reverse_indices: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        """
        Combine the outputs of the routers into the final outputs
        """
        assert len(inputs) == self.route_num and len(reverse_indices) == self.route_num
        route_results = torch.zeros(loads, *self.route_shape)
        for i in range(self.route_num):
            if reverse_indices[i].numel() > 0:
                indices = (
                    reverse_indices[i]
                    .repeat(1, self.route_size)
                    .view(-1, *self.route_shape)
                )
                if self.reduction == "add":
                    torch.scatter_add(route_results, 0, indices, inputs[i])
                else:
                    raise NotImplementedError(
                        f"Reduction method {self.reduction} is not supported"
                    )
        return route_results
