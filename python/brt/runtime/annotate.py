# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List

import torch
import torch.nn as nn
from brt.runtime.grid_tensor import GridTensor, init_grid_tensor
from brt.trace.leaf_node import register_leaf_node


@register_leaf_node
class Annotator(nn.Module):
    def __init__(
        self, dims: List[int], cell_shape: List[int] = None, gridding: bool = True
    ):
        super().__init__()
        assert isinstance(dims, list)
        dims = sorted(dims)
        self.dims = dims
        if cell_shape is not None:
            assert isinstance(cell_shape, list)
        self.cell_shape = cell_shape
        self.gridding = gridding

    def forward(self, t: torch.Tensor):
        assert self.dims[-1] < len(t.shape)
        if self.cell_shape is None:
            cell_shape = [t.shape[i] for i in range(len(t.shape)) if i not in self.dims]
        else:
            cell_shape = self.cell_shape
        transposed_dims = self.dims + [
            i for i in range(len(t.shape)) if i not in self.dims
        ]
        transposed_tensor = t.permute(*transposed_dims).contiguous()
        reshaped_tensor = transposed_tensor.reshape(-1, *cell_shape)
        if self.gridding:
            if isinstance(reshaped_tensor, GridTensor):
                initialized_tensor = reshaped_tensor.pack(None, None)
            else:
                initialized_tensor = init_grid_tensor(reshaped_tensor, [None], [None])
        else:
            initialized_tensor = reshaped_tensor
        return initialized_tensor
