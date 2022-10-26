# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import brt._C.distributed as C_dist
import torch
import torch.distributed as dist


def init_nccl(group: dist.ProcessGroup):
    world_size = dist.get_world_size(group)
    world_rank = dist.get_rank(group)
    unique_id = C_dist.make_nccl_unique_id(world_rank)
    dist.broadcast(unique_id, 0, group)
    C_dist.init_nccl(unique_id, world_rank, world_size)


def asymmetry_a2a(in_data: torch.Tensor, in_loads: torch.Tensor, locality_aware=False):
    return C_dist.asymmetry_all_to_all(in_data, in_loads, locality_aware)



def locality_aware_a2a(in_data: torch.Tensor, in_loads: torch.Tensor):
    out_data, out_loads, reorder_indices = C_dist.asymmetry_all_to_all(
        in_data, in_loads, True
    )
    return out_data, out_loads, reorder_indices
