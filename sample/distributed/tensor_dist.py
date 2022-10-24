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
brt_dist.init_nccl(group)

grain_size = 768
capacity = 1024


tensor = torch.arange(
    local_rank * world_size * capacity * grain_size,
    (local_rank + 1) * world_size * capacity * grain_size,
    device=device,
).reshape(-1, grain_size)
loads = torch.randint(
    1, capacity + 1, (world_size,), dtype=torch.int32, device=device
)


# print(tensor)
# print(loads)
out_data, out_loads = brt_dist.asymmetry_all_to_all(tensor, loads)
# print(out_data)
# print(out_loads)

timer = CUDATimer(repeat=2)
timer.execute(
    lambda: brt_dist.asymmetry_all_to_all(tensor, loads), "brt.asymmetry_all_to_all"
)


def torch_asymmetry_all_to_all(tensor):
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor)
    return output


timer.execute(lambda: torch_asymmetry_all_to_all(tensor), "dist.all_to_all_single")
