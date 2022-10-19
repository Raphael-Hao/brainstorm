# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import brt._C.distributed as C_dist
import torch.distributed as dist


def init_nccl(group: dist.ProcessGroup, event_num=1):
    world_rank = dist.get_world_size(group)
    world_size = dist.get_rank(group)
    unique_id = C_dist.make_nccl_unique_id(world_rank)
    if world_rank == 0:
        dist.broadcast(unique_id, 0, group)
    C_dist.init_nccl(unique_id, world_rank, world_size, event_num)


def asymmetry_all_to_all(in_data, loads):
    out_data = C_dist.asymmetry_all_to_all(in_data, loads)
    return out_data