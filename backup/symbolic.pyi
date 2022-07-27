from typing import List, Tuple

import torch


def symbolic_scatter_route(
    inputs: torch.Tensor, route_num: int
) -> List[torch.Tensor]: ...
def symbolic_gather_route(
    inputs: List[torch.Tensor],
    route_num: int,
) -> torch.Tensor: ...
def symbolic_joint_scatter_route(
    inputs: List[torch.Tensor], route_num: int
) -> List[List[torch.Tensor]]: ...
def symbolic_joint_gather_route(
    inputs: List[List[torch.Tensor]],
    route_num: int,
) -> List[torch.Tensor]: ...
