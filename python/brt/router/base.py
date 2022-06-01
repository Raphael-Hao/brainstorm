# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union

import torch
import torch.nn as nn
from brt.primitive import router

from .symbolic import symbolic_tag_route


@router
class BaseRouter(nn.Module):
    def __init__(
        self,
        route_num: int,
    ):
        """_summary_

        Args:
            route_num (int): number of src or dst for routing
            gran_dim (_type_, optional): routing granularity. should be a int or a list of int.
        """
        super().__init__()
        self.route_num = route_num
        self.active_counter = 0

    def route(self, *inputs):
        raise NotImplementedError

    def symbolic_route(self, *inputs):
        raise NotImplementedError


@router
class TagRouter(BaseRouter):
    def __init__(self):
        super().__init__(route_num=1)

    def route(
        self, inputs: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        loads = inputs.size(0)
        tags = torch.arange(0, loads, dtype=torch.long, device=inputs.device)
        return inputs, tags, loads

    def symbolic_route(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return symbolic_tag_route(inputs)
