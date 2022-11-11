#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /distributed.pyi
# \brief:
# Author: raphael hao
from typing import Tuple, overload, Literal, List
import torch

def make_nccl_unique_id(world_rank: int) -> torch.Tensor: ...
def init_nccl(unique_id: torch.Tensor, world_rank: int, world_size: int) -> None: ...
def locality_reorder(
    loads: torch.Tensor, world_size: int
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def group_locality_reorder(
    loads: torch.Tensor, world_size: int, group_size: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def exchange(in_data: torch.Tensor, reorder_indices: torch.Tensor) -> torch.Tensor: ...
def batched_exchange(
    in_datas: List[torch.Tensor], reorder_indices: torch.Tensor
) -> List[torch.Tensor]: ...
def reverse_exchange(
    in_data: torch.Tensor, reorder_indices: torch.Tensor
) -> torch.Tensor: ...
def batched_reverse_exchange(
    in_datas: List[torch.Tensor], reorder_indices: torch.Tensor
) -> List[torch.Tensor]: ...
@overload
def asymmetry_all_to_all(
    in_data: torch.Tensor, send_sizes: torch.Tensor, locality_aware: Literal[False]
) -> Tuple[torch.Tensor, torch.Tensor]: ...
@overload
def asymmetry_all_to_all(
    in_data: torch.Tensor, send_sizes: torch.Tensor, locality_aware: Literal[True]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@overload
def group_asymmetry_all_to_all(
    in_data: torch.Tensor, send_sizes: torch.Tensor, locality_aware: Literal[False]
) -> Tuple[torch.Tensor, torch.Tensor]: ...
@overload
def group_asymmetry_all_to_all(
    in_data: torch.Tensor, send_sizes: torch.Tensor, locality_aware: Literal[True]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def batched_group_asymmetry_all_to_all(
    in_datas: List[torch.Tensor],
    send_sizes: torch.Tensor,
    locality_aware: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]: ...
def size_known_group_asymmetry_all_to_all(
    in_data: torch.Tensor,
    send_sizes: torch.Tensor,
    recv_sizes: torch.Tensor,
) -> torch.Tensor: ...
def batched_size_known_group_asymmetry_all_to_all(
    in_datas: List[torch.Tensor],
    send_sizes: torch.Tensor,
    recv_sizes: torch.Tensor,
) -> List[torch.Tensor]: ...