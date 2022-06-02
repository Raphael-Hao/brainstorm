from typing import List

import numpy as np
import torch

from .default_dispatcher import DefaultDispatcher


class ResidualDispatcher(DefaultDispatcher):
    """
    Default dispatcher implement by Brainstorm
    The default dispatcher will dispatch the inputs to the routers just according to the ground-truth routing indices.
    """

    def __init__(self, route_num, transform=True, reduction="add", residual_route=0):
        super().__init__(route_num, transform, reduction)
        self.residual_route = residual_route

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
        residual_indices = (route_indices.sum(dim=1) == 0).long().to(inputs.device)
        for i in range(self.route_num):
            tags = torch.nonzero(
                route_indices[:, i] + residual_indices
                if i == self.residual_route
                else route_indices[:, i]
            )
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
