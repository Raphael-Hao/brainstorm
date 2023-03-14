# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, overload, Literal, List, Union

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
    is_tag_index=False,
    load_on_cpu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def dispatch_with_indices(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    gates: torch.Tensor = None,
    tag_generating: Literal = True,
    tags: torch.Tensor = None,
    max_path_padding: bool = False,
    max_path_load: int = None,
    is_1d_routing: bool = True,
    is_tag_index: bool = False,
) -> Tuple[torch.Tensor,torch.Tensor]: ...
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
    is_tag_index: bool = False,
) -> torch.Tensor: ...
def combine_with_src_indices(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    auto_pad: bool = False,
    gates: torch.Tensor = None,
    out_data: torch.Tensor = None,
) -> torch.Tensor: ...
@overload
def split_fused_cells_to_paths(
    in_data: torch.Tensor,
    loads: torch.Tensor,
    max_path_padding: bool = False,
    is_load_split: Literal[False] = False,
    is_tag_split: Literal[False] = False,
    tags: torch.Tensor = None,
) -> Tuple[List[torch.Tensor]]: ...
@overload
def split_fused_cells_to_paths(
    in_data: torch.Tensor,
    loads: torch.Tensor,
    max_path_padding: bool = False,
    is_load_split: Literal[True] = False,
    is_tag_split: Literal[False] = False,
    tags: torch.Tensor = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]: ...
@overload
def split_fused_cells_to_paths(
    in_data: torch.Tensor,
    loads: torch.Tensor,
    max_path_padding: bool = False,
    is_load_split: bool = False,
    is_tag_split: Literal[True] = False,
    tags: torch.Tensor = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: ...
@overload
def fuse_split_cells_from_paths(
    in_data: List[torch.Tensor],
    is_load_fuse: Literal[True],
    is_tag_fuse: Literal[False],
    loads: List[torch.Tensor] = None,
    tags: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
@overload
def fuse_split_cells_from_paths(
    in_data: List[torch.Tensor],
    is_load_fuse: Literal[False],
    is_tag_fuse: Literal[True],
    loads: List[torch.Tensor] = None,
    tags: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@overload
def fuse_split_cells_from_paths(
    in_data: List[torch.Tensor],
    is_load_fuse: Literal[False],
    is_tag_fuse: Literal[False],
    loads: List[torch.Tensor] = None,
    tags: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor]: ...
def fuse_split_cells_from_paths(
    in_data: List[torch.Tensor],
    is_load_fuse: bool = False,
    is_tag_fuse: bool = False,
    loads: List[torch.Tensor] = None,
    tags: List[torch.Tensor] = None,
) -> Union[
    Tuple[torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]: ...
def combine_with_indices_and_loads(
    in_data: torch.Tensor,
    route_indices: torch.Tensor,
    loads: torch.Tensor,
    gates: torch.Tensor = None,
    out_data: torch.Tensor = None,
    tag_generating: bool = False,
    tags: torch.Tensor = None,
    max_path_padding: bool = False,
    ever_padded=True,
    is_tag_index: bool = False,
) -> torch.Tensor: ...

