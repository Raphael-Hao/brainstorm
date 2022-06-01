from typing import List, Tuple

import torch


def symbolic_tag_route(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
def symbolic_scatter_route(
    inputs: torch.Tensor, route_num: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]: ...
def symbolic_gather_route(
    inputs: List[torch.Tensor],
    tags: List[torch.Tensor],
    loads: int,
    route_num: int,
) -> torch.Tensor: ...
