#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /distributed.pyi
# \brief:
# Author: raphael hao

import torch

def make_nccl_unique_id(world_rank: int) -> torch.Tensor: ...
def init_nccl(
    unique_id: torch.Tensor, world_rank: int, world_size: int, event_num=1
) -> None: ...
def asymmetry_all_to_all(in_data, send_sizes, recv_sizes) -> torch.Tensor: ...
