# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import torch


def generate_global_route_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def generate_local_route_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def route_data_with_local_indices(
    in_data: torch.Tensor,
    out_data: torch.Tensor,
    gates: torch.Tensor,
    route_indices: torch.Tensor,
    dst_loads: torch.Tensor,
) -> torch.Tensor: ...
