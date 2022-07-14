# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple

import torch


def generate_global_dst_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def generate_dst_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def generate_src_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def convert_index_format(
    origin_indices: torch.Tensor,
    loads: torch.Tensor,
    new_index_fmt_id: int,  # 0 for src_index or 1 for dst_index
) -> torch.Tensor: ...
def route_with_dst_indices(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    dst_loads: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor: ...
def route_back_with_dst_indices(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    dst_loads: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor: ...
