# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist
import brt.runtime.distributed as brt_dist

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
group = dist.group.WORLD
brt_dist.init_nccl(group)
tensor = torch.arange(local_rank * 10, (local_rank + 1) * 10, device=device).reshape(
    -1, 1
)
loads = torch.randint(1, 5, (world_size,), dtype=torch.int32, device=device)
print(tensor)
print(loads)
out_data, out_loads = brt_dist.asymmetry_all_to_all(tensor, loads)
print(out_data)
print(out_loads)
