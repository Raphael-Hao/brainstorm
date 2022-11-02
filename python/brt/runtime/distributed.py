# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Tuple, overload, Literal

import brt._C.distributed as C_dist
import torch
import torch.distributed as dist


def init_nccl(group: dist.ProcessGroup):
    world_size = dist.get_world_size(group)
    world_rank = dist.get_rank(group)
    unique_id = C_dist.make_nccl_unique_id(world_rank)
    dist.broadcast(unique_id, 0, group)
    C_dist.init_nccl(unique_id, world_rank, world_size)


@overload
def asymmetry_a2a(
    in_data: torch.Tensor, in_loads: torch.Tensor, locality_aware: Literal[False]
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@overload
def asymmetry_a2a(
    in_data: torch.Tensor, in_loads: torch.Tensor, locality_aware: Literal[True]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...


def asymmetry_a2a(
    in_data: torch.Tensor, in_loads: torch.Tensor, locality_aware: bool = False
):
    return C_dist.asymmetry_all_to_all(in_data, in_loads, locality_aware)

@overload
def group_asymmetry_a2a(
    in_data: torch.Tensor, in_loads: torch.Tensor, locality_aware: Literal[False]
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@overload
def group_asymmetry_a2a(
    in_data: torch.Tensor, in_loads: torch.Tensor, locality_aware: Literal[True]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...


def group_asymmetry_a2a(
    in_data: torch.Tensor, in_loads: torch.Tensor, locality_aware: bool = False
):
    return C_dist.group_asymmetry_all_to_all(in_data, in_loads, locality_aware)
