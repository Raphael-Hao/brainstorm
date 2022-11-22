# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist
import brt.runtime.distributed as brt_dist
from brt.runtime.benchmark import CUDATimer

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
group = dist.group.WORLD
brt_dist.is_nccl_activated(group)

tensor = torch.arange(
    local_rank * world_size, (local_rank + 1) * world_size, device=device
)
if local_rank == 0:
    indices = torch.randperm(world_size, dtype=torch.int32, device=device)
    print(indices)
else:
    indices = torch.empty(world_size, dtype=torch.int32, device=device)
dist.broadcast(indices, 0, group=group)

tensor = brt_dist.exchange(tensor, indices)
print(tensor)

capacity = 2

tensors = [
    torch.arange(
        local_rank * world_size * capacity + i * world_size,
        local_rank * world_size * capacity + (i + 1) * world_size,
        device=device,
    )
    for i in range(capacity)
]
if local_rank == 0:
    indices = torch.randperm(world_size, dtype=torch.int32, device=device)
    print(indices)
else:
    indices = torch.empty(world_size, dtype=torch.int32, device=device)
dist.broadcast(indices, 0, group=group)

tensors = brt_dist.batched_exchange(tensors, indices)
tensors = brt_dist.batched_reverse_exchange(tensors, indices)
print(tensors)


timer = CUDATimer(repeat=2, root=local_rank)
# timer.execute(lambda: brt_dist.asymmetry_a2a(tensor, loads), "brt.asymmetry_all_to_all")


# def torch_symmetry_a2a(tensor, loads):
#     out_loads = torch.empty_like(loads)
#     dist.all_to_all_single(out_loads, loads)
#     # torch.cuda.synchronize()
#     output = torch.empty_like(tensor)
#     dist.all_to_all_single(output, tensor)
#     return output


# timer.execute(lambda: torch_symmetry_a2a(tensor, loads), "dist.all_to_all_single")
