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
        gran_dim: Union[int, List[int]],
    ):
        """_summary_

        Args:
            route_num (int): number of src or dst for routing
            gran_dim (_type_, optional): routing granularity. should be a int or a list of int.
        """
        super().__init__()
        self.route_num = route_num
        self.gran_dim = gran_dim
        self.active_counter = 0

    # def __init_subclass__(cls):
    #     if getattr(cls, "_traced", False) is False:
    #         cls = router(cls)

    def route(self, *inputs):
        raise NotImplementedError

    def symbolic_route(self, *inputs):
        raise NotImplementedError


@router
class TagRouter(BaseRouter):
    def __init__(self, gran_dim: Union[int, List[int]]):
        super().__init__(route_num=1, gran_dim=gran_dim)

    def route(
        self, inputs: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        loads = inputs.size(0)
        indices = torch.arange(0, loads, dtype=torch.long, device=inputs.device)
        return inputs, indices, loads

    def symbolic_route(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return symbolic_tag_route(inputs)


class SparseRouter(BaseRouter):
    def __init__(self, gran_dim: Union[int, List[int]]):
        super().__init__(route_num=1, gran_dim=gran_dim)

    def route(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        return inputs
