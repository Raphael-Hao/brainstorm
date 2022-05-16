# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import torch
import torch.nn as nn
from brt.common import find_lib_path
from brt.primitive import router

torch.ops.load_library(find_lib_path("libbrt_torchscript.so")[0])

@router
class BaseRouter(nn.Module):
    def __init__(self, route_num: int, gran_dim: int = None, dtype=None):
        """_summary_

        Args:
            route_num (int): number of src or dst for routing
            gran_dim (_type_, optional): routing granularity. Defaults to None.
                                          if None, routers only accept list of tensors
        """
        super().__init__()
        self.route_num = route_num
        self.gran_dim = gran_dim
        self.dtype = torch.float32 if dtype is None else dtype
        self.active_counter = 0

    def route(self, *args, **kwargs):
        raise NotImplementedError

    def symbolic_route(self, *args, **kwargs):
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
