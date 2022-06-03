from typing import List, Tuple

import torch
from brt.primitive import router

from .base import BaseRouter
from .symbolic import symbolic_tag_route

__all__ = ["TagRouter"]


@router
class TagRouter(BaseRouter):
    def __init__(self):
        super().__init__(route_num=1)

    def route(
        self, inputs: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], int]:
        tags = torch.arange(
            0, inputs.size(0), dtype=torch.long, device=inputs.device
        ).view(-1, 1)
        return inputs, tags

    def symbolic_route(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return symbolic_tag_route(inputs)
