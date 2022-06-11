# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch
from brt.routers.proto_tensor import ProtoTensor

from .base import Dispatcher


class DefaultDispatcher(Dispatcher):
    """
    Default dispatcher implement by Brainstorm
    The default dispatcher will dispatch the inputs to the routers just according to the ground-truth routing indices.
    """

    def __init__(self, route_num, transform=True, reduction="add", sparse=True):
        """currently we only support add due to the scatter_reduce of torch
        is in development and will be released in 1.12, After that we will
        support "sum", "prod", "mean", "amax", "amin, see details in
        https://github.com/pytorch/pytorch/issues/74770
        """
        super().__init__(route_num, transform, reduction)
        self.sparse = sparse

    def dispatch(
        self,
        inputs: ProtoTensor,
        route_indices: torch.Tensor,
        gates: torch.Tensor,
    ) -> List[ProtoTensor]:
        """
        Dispatch the inputs into the the list of torch.Tensor with indices
        """
        route_data = inputs.data
        route_tag = inputs.tag
        load = inputs.load
        route_shape = list(route_data.shape[1:])
        route_size = np.prod(route_shape)

        results = [
            ProtoTensor(
                data=torch.zeros(
                    0, *route_shape, dtype=route_data.dtype, device=route_data.device
                ),
                tag=torch.zeros(0, 1, dtype=torch.int64, device=route_data.device),
                load=load,
            )
            for _ in range(self.route_num)
        ]
        for i in range(self.route_num):
            tag_indices = torch.nonzero(route_indices[:, i].view(-1)).to(
                route_data.device
            )
            if tag_indices.numel() > 0:
                results[i].tag = torch.gather(route_tag, 0, tag_indices)
                data_indices = tag_indices.repeat(1, route_size).view(-1, *route_shape)
                if self.transform:
                    gate = gates[:, i].reshape(
                        (route_data.size(0),) + (1,) * len(route_shape)
                    )
                    results[i].data = torch.gather(route_data * gate, 0, data_indices)
                else:
                    results[i].data = torch.gather(route_data, 0, data_indices)
            if self.sparse:
                results[i].load = tag_indices.numel()
        return results

    def combine(self, inputs: List[ProtoTensor]) -> ProtoTensor:
        """
        Combine the outputs of the routers into the final outputs
        """
        assert len(inputs) == self.route_num
        route_datas = torch.cat([_input.data for _input in inputs], dim=0)
        route_tags = torch.cat([_input.tag for _input in inputs], dim=0)
        load = np.max([_input.load for _input in inputs])
        route_shape = list(route_datas.shape[1:])
        route_size = np.prod(route_shape)
        if route_datas.numel() == 0:
            results_data = torch.zeros(
                0, *route_shape, dtype=route_datas.dtype, device=route_datas.device
            )
            return ProtoTensor(results_data, route_tags, load)
        if self.sparse:
            result_tag, inverse = torch.unique(
                route_tags, sorted=True, return_inverse=True
            )
            route_indices = inverse.repeat(1, route_size).view(-1, *route_shape)
            load = result_tag.size(0)
        else:
            route_indices = route_tags.repeat(1, route_size).view(-1, *route_shape)
            result_tag = torch.arange(
                0, load, dtype=torch.int64, device=route_datas.device
            )
        result_data = torch.zeros(
            load, *route_shape, dtype=route_datas.dtype, device=route_datas.device
        ).scatter_(0, route_indices, route_datas, reduce=self.reduction)
        # results_data = torch.scatter_reduce(route_datas, 0, route_indices, self.reduction)
        return ProtoTensor(result_data, route_tags, load)
