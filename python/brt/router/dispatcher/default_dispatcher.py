# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .base import Dispatcher


class DefaultDispatcher(Dispatcher):
    """
    Default dispatcher implement by Brainstorm
    The default dispatcher will dispatch the inputs to the routers just according to the ground-truth routing indices.
    """

    def __init__(self, route_num):
        super().__init__(route_num)

    def dispatch(self, inputs, route_indices):
        """
        Dispatch the inputs into the the list of torch.Tensor with indices
        """
        route_results = [None] * self.route_num
        reverse_indices = [None] * self.route_num
        for i in range(self.route_num):
            indices = torch.nonzero(route_indices == i)[0]
            if len(indices) > 0:
                tmp_results = [inputs[j] for j in indices]
                # TODO: current only support tensors with same shape
                route_results[i] = torch.stack(tmp_results)
                reverse_indices[i] = indices
        return route_results, reverse_indices

    def combine(
        self,
        reverse_indices,
        reverse_shape,
        *inputs,
    ):
        """
        Combine the outputs of the routers into the final outputs
        """
        assert (
            len(inputs) == self._route_num and len(reverse_indices) == self._route_num
        )
        if isinstance(reverse_shape, int):
            route_size = reverse_shape
        elif isinstance(reverse_shape, torch.Size):
            route_size = reverse_shape[0]
        else:
            raise ValueError("origin_shape must be a int or torch.Size")
        route_results = [[] for _ in range(route_size)]
        for i in range(self._route_num):
            if reverse_indices[i] is not None:
                for j in range(len(reverse_indices[i])):
                    route_results[reverse_indices[i][j]] = inputs[i][j]
        if isinstance(reverse_shape, int):
            return route_results
        else:
            route_results = torch.stack(route_results).view(reverse_shape)
        return route_results
