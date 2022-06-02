from typing import List, Tuple

import torch
from brt.primitive import router

from .base import BaseRouter
from .symbolic import symbolic_tag_route


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
