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


def _is_initialized():
    return global_info.initialized


def is_nccl_activated(group: dist.ProcessGroup = None) -> bool:
    if not dist.is_initialized():
        return False
    if not _is_initialized():
        global_info.initialized = True
        global_info.group = group or dist.group.WORLD
        global_info.world_size = dist.get_world_size(group)
        global_info.world_rank = dist.get_rank(group)
        unique_id = C_dist.make_nccl_unique_id(global_info.world_rank)
        dist.broadcast(unique_id, 0, global_info.group)
        C_dist.init_nccl(unique_id, global_info.world_rank, global_info.world_size)
    return True


def get_reorder_indices():
    return global_info.reorder_indices


def set_reorder_indices(reorder_indices: torch.Tensor):
    global_info.reorder_indices = reorder_indices


def exchange(in_data: torch.Tensor, reorder_indices: torch.Tensor):
    if is_nccl_activated():
        return C_dist.exchange(in_data, reorder_indices)
    else:
        return in_data


def batched_exchange(in_datas: List[torch.Tensor], reorder_indices: torch.Tensor):
    if is_nccl_activated():
        return C_dist.batched_exchange(in_datas, reorder_indices)
    else:
        return in_datas


def reverse_exchange(in_data: torch.Tensor, reorder_indices: torch.Tensor):
    if is_nccl_activated():
        return C_dist.reverse_exchange(in_data, reorder_indices)
    return in_data


def batched_reverse_exchange(
    in_datas: List[torch.Tensor], reorder_indices: torch.Tensor
):
    if is_nccl_activated():
        return C_dist.batched_reverse_exchange(in_datas, reorder_indices)
    else:
        return in_datas


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
    if is_nccl_activated():
        return C_dist.asymmetry_all_to_all(in_data, in_loads, locality_aware)
    else:
        assert locality_aware is False
        return in_data, in_loads


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
    if is_nccl_activated():
        return C_dist.group_asymmetry_all_to_all(in_data, in_loads, locality_aware)
    else:
        assert locality_aware is False
        return in_data, in_loads


@overload
def batched_group_asymmetry_a2a(
    in_datas: List[torch.Tensor], in_loads: torch.Tensor, locality_aware: Literal[False]
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    ...


@overload
def batched_group_asymmetry_a2a(
    in_datas: List[torch.Tensor], in_loads: torch.Tensor, locality_aware: Literal[True]
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    ...


def batched_group_asymmetry_a2a(
    in_datas: List[torch.Tensor], in_loads: torch.Tensor, locality_aware: bool = False
):
    if is_nccl_activated():
        (
            out_datas,
            out_loads,
            reorder_indices,
        ) = C_dist.batched_group_asymmetry_all_to_all(
            in_datas, in_loads, locality_aware
        )
        if locality_aware:
            return out_datas, out_loads, reorder_indices
        else:
            return out_datas, out_loads
    else:
        assert locality_aware is False
        return in_datas, in_loads


def size_known_group_asymmetry_all_to_all(
    in_data: torch.Tensor, in_loads: torch.Tensor, out_loads: torch.Tensor
):
    if is_nccl_activated():
        return C_dist.size_known_group_asymmetry_all_to_all(
            in_data, in_loads, out_loads
        )
    else:
        return in_data


def batched_size_known_group_asymmetry_all_to_all(
    in_datas: List[torch.Tensor], in_loads: torch.Tensor, out_loads: torch.Tensor
):
    if is_nccl_activated():
        return C_dist.batched_size_known_group_asymmetry_all_to_all(
            in_datas, in_loads, out_loads
        )
    else:
        return in_datas


def group_sparse_all_to_all(in_data: torch.Tensor, in_loads: torch.Tensor):
    if is_nccl_activated():
        return C_dist.group_sparse_all_to_all(in_data, in_loads)
    else:
        return in_data


def size_known_group_sparse_all_to_all(
    in_data: torch.Tensor, in_loads: torch.Tensor, out_loads: torch.Tensor
):
    if is_nccl_activated():
        return C_dist.size_known_group_sparse_all_to_all(in_data, in_loads, out_loads)
    else:
        return in_data
