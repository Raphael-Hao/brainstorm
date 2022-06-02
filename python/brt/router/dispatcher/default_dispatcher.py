# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import numpy as np
import torch

from .base import Dispatcher


class DefaultDispatcher(Dispatcher):
    """
    Default dispatcher implement by Brainstorm
    The default dispatcher will dispatch the inputs to the routers just according to the ground-truth routing indices.
    """

    def __init__(self, route_num, transform=True, reduction="add"):
        super().__init__(route_num, transform, reduction)

    def dispatch(
        self, inputs: torch.Tensor, route_indices: torch.Tensor, gates: torch.Tensor
    ):
        """
        Dispatch the inputs into the the list of torch.Tensor with indices
        """
        route_shape = list(inputs.shape[1:])
        route_size = np.prod(route_shape)
        route_results = [torch.zeros(0, *route_shape)] * self.route_num
        route_tags = [
            torch.zeros(0, 1, dtype=torch.int64, device=inputs.device)
        ] * self.route_num
        for i in range(self.route_num):
            tags = torch.nonzero(route_indices[:, i].view(-1))
            if tags.numel() > 0:
                route_tags[i] = tags
                tags = tags.repeat(1, route_size).view(-1, *route_shape)
                if self.transform:
                    gate = gates[:, i].reshape(
                        (inputs.size(0),) + (1,) * len(route_shape)
                    )
                    route_results[i] = torch.gather(inputs * gate, 0, tags)
                else:
                    route_results[i] = torch.gather(inputs, 0, tags)
        return route_results, route_tags

    def combine(
        self,
        inputs: List[torch.Tensor],
        tags: List[torch.Tensor],
        loads: int,
    ) -> torch.Tensor:
        """
        Combine the outputs of the routers into the final outputs
        """
        assert len(inputs) == self.route_num and len(tags) == self.route_num
        route_shape = list(inputs[0].shape[1:])
        inputs = torch.cat(inputs, dim=0)
        tags = (
            torch.cat(tags, dim=0)
            .repeat(1, np.prod(route_shape))
            .view(-1, *route_shape)
        )
        if inputs.numel() == 0:
            route_results = torch.zeros(0, *route_shape)
        else:
            route_results = torch.zeros(loads, *route_shape)
            if self.reduction == "add":
                route_results = torch.scatter_add(route_results, 0, tags, inputs)
            else:
                raise NotImplementedError(
                    f"Reduction method {self.reduction} is not supported"
                )
        return route_results
