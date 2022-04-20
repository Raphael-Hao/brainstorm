# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union, List
import torch.nn as nn
import torch


class Router(nn.Module):
    _router_tag = True
    def __init__(self, route_num: int, grain_dim: int = None, dtype=None):
        """_summary_

        Args:
            route_num (int): number of src or dst for routing
            grain_dim (_type_, optional): routing granularity. Defaults to None.
                                          if None, routers only accept list of tensors
        """
        super().__init__()
        self.route_num = route_num
        self.grain_dim = grain_dim
        self.dtype = torch.float32 if dtype is None else dtype
        self.active_counter = 0

    @torch.jit.ignore
    def forward(self, *args, **kwargs):
        return self.route(*args, **kwargs)

    def route(self, *inputs, **kwargs):
        raise NotImplementedError

    def inspect_inputs(self, inputs: Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(inputs, torch.Tensor):
            inputs_shape = inputs.shape
            inputs_size = inputs.shape[0]
            tensor_input = True
        elif isinstance(inputs, list):
            inputs_size = len(inputs)
            inputs_shape = inputs_size
            tensor_input = False
        else:
            raise ValueError(f"Inputs: {inputs}  must be a list of tensor or a tensor")
        return inputs_size, inputs_shape, tensor_input

    def record(self):
        raise NotImplementedError

    def register_router(self):
        pass
