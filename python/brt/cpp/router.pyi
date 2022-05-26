# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import torch


def generate_indices_with_load_map(
    one_hot_mask: torch.Tensor,
    route_indices: torch.Tensor,
    branch_loads: torch.Tensor,
    branch_start_indices: torch.Tensor,
    supported_capacities: torch.Tensor,
    sample_num: int,
    branch_num: int,
    supported_capacity_num: int,
) -> None: ...

