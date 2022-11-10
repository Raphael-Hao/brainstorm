# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Tuple, overload, Literal, List

import brt._C.distributed as C_dist
import torch
import torch.distributed as dist


class global_info:
    initialized = False
    reorder_indices = None
    group = None
    world_size = None
    world_rank = None


def is_initialized():
    return global_info.initialized


def init_nccl(group: dist.ProcessGroup = None):
    if is_initialized():
        return
    global_info.initialized = True
    global_info.group = group or dist.group.WORLD
    global_info.world_size = dist.get_world_size(group)
    global_info.world_rank = dist.get_rank(group)
    unique_id = C_dist.make_nccl_unique_id(global_info.world_rank)
    dist.broadcast(unique_id, 0, global_info.group)
    C_dist.init_nccl(unique_id, global_info.world_rank, global_info.world_size)


def get_reorder_indices():
    return global_info.reorder_indices


def set_reorder_indices(reorder_indices: torch.Tensor):
    global_info.reorder_indices = reorder_indices


def exchange(in_data: torch.Tensor, reorder_indices: torch.Tensor):
    init_nccl()
    return C_dist.exchange(in_data, reorder_indices)


def batch_exchange(in_datas: List[torch.Tensor], reorder_indices: torch.Tensor):
    init_nccl()
    return C_dist.batch_exchange(in_datas, reorder_indices)


def reverse_exchange(in_data: torch.Tensor, reorder_indices: torch.Tensor):
    init_nccl()
    return C_dist.reverse_exchange(in_data, reorder_indices)


def batch_reverse_exchange(in_datas: List[torch.Tensor], reorder_indices: torch.Tensor):
    init_nccl()
    return C_dist.batch_reverse_exchange(in_datas, reorder_indices)


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
    init_nccl()
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
    init_nccl()
    return C_dist.group_asymmetry_all_to_all(in_data, in_loads, locality_aware)


@overload
def batch_group_asymmetry_a2a(
    in_datas: List[torch.Tensor], in_loads: torch.Tensor, locality_aware: Literal[False]
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    ...


@overload
def batch_group_asymmetry_a2a(
    in_datas: List[torch.Tensor], in_loads: torch.Tensor, locality_aware: Literal[True]
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    ...


def batch_group_asymmetry_a2a(
    in_datas: List[torch.Tensor], in_loads: torch.Tensor, locality_aware: bool = False
):
    init_nccl()
    out_datas, out_loads, reorder_indices = C_dist.batch_group_asymmetry_all_to_all(
        in_datas, in_loads, locality_aware
    )
    if locality_aware:
        return out_datas, out_loads, reorder_indices
    else:
        return out_datas, out_loads


def size_known_group_asymmetry_all_to_all(
    in_data: torch.Tensor, in_loads: torch.Tensor, out_loads: torch.Tensor
):
    init_nccl()
    return C_dist.size_known_group_asymmetry_all_to_all(in_data, in_loads, out_loads)


def batch_size_known_group_asymmetry_all_to_all(
    in_datas: List[torch.Tensor], in_loads: torch.Tensor, out_loads: torch.Tensor
):
    init_nccl()
    return C_dist.batch_size_known_group_asymmetry_all_to_all(
        in_datas, in_loads, out_loads
    )
