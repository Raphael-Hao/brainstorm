# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import torch

def convert_index_format(
    origin_indices: torch.Tensor,
    loads: torch.Tensor,
    new_index_fmt_id: int,  # 0 for src_index or 1 for dst_index
) -> torch.Tensor: ...
def generate_indices_and_loads(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor = None,
    capacity_padding: bool = False,
    is_dst_index=True,
    load_on_cpu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def dispatch_with_indices_and_loads(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    gates: torch.Tensor = None,
    tag_generating: bool = False,
    tags: torch.Tensor = None,
    max_path_padding: bool = False,
    max_path_load: int = None,
    is_1d_routing: bool = True,
    is_dst_index: bool = True,
) -> torch.Tensor: ...
def dispatch_with_dst_indices_1d(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    auto_pad: bool = False,
    gates: torch.Tensor = None,
) -> torch.Tensor: ...
def padded_dispatch_with_dst_indices_1d(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    pad_size: int,
    gates: torch.Tensor = None,
) -> torch.Tensor: ...
def dispatch_with_dst_indices_2d(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    auto_pad: bool = False,
) -> torch.Tensor: ...
def combine_with_src_indices(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    auto_pad: bool = False,
    gates: torch.Tensor = None,
    out_data: torch.Tensor = None,
) -> torch.Tensor: ...
