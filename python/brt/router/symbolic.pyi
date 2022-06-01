from typing import List, Tuple

import torch

def symbolic_tag_route(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...
def symbolic_scatter_route(
    inputs: torch.Tensor, router_kind: int, route_num: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]: ...
def symbolic_gather_route(
    inputs: List[torch.Tensor],
    reverse_indices: List[torch.Tensor],
    loads: int,
    router_kind: int,
    route_num: int,
) -> torch.Tensor: ...
