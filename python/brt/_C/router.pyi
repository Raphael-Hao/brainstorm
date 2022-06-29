# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import torch


def generate_global_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def generate_dst_indices(
    hot_mask: torch.Tensor,
) -> List[torch.Tensor]: ...
def generate_local_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def route_with_local_indices(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    dst_loads: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor: ...
def route_back_with_local_indices(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    dst_loads: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor: ...
