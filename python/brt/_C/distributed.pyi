#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /distributed.pyi
# \brief:
# Author: raphael hao
from typing import Tuple
import torch

def make_nccl_unique_id(world_rank: int) -> torch.Tensor: ...
def init_nccl(
    unique_id: torch.Tensor, world_rank: int, world_size: int, event_num=1
) -> None: ...
def locality_reorder(loads, world_size) -> Tuple[torch.Tensor, torch.Tensor]: ...
def asymmetry_all_to_all(
    in_data, send_sizes, locality=False
) -> Tuple[torch.Tensor, torch.Tensor]: ...
