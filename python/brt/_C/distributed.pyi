#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /distributed.pyi
# \brief:
# Author: raphael hao
from typing import Tuple, overload
import torch

def make_nccl_unique_id(world_rank: int) -> torch.Tensor: ...
def init_nccl(
    unique_id: torch.Tensor, world_rank: int, world_size: int
) -> None: ...
def locality_reorder(
    loads: torch.Tensor, world_size: int
) -> Tuple[torch.Tensor, torch.Tensor]: ...
@overload
def asymmetry_all_to_all(
    in_data: torch.Tensor, send_sizes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]: ...
@overload
def asymmetry_all_to_all(
    in_data: torch.Tensor, send_sizes: torch.Tensor, locality: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
