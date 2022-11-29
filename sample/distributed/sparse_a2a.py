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

grain_size = 1
group_size = 2
if local_rank == 0:
    loads = torch.tensor([3, 2, 4, 4, 2, 4, 5, 2], dtype=torch.int32, device=device)
else:
    loads = torch.tensor([1, 22, 3, 4, 3, 55, 222, 1], dtype=torch.int32, device=device)

tensor = torch.randn(loads.sum().item(), grain_size, device=device)
print(f"input tensor: {tensor}")

out_data, out_loads, in_loads = brt_dist.group_sparse_all_to_all(tensor, loads)

print(f"output tensor: {out_data}")
print(f"output loads: {out_loads}")
print(f"input loads: {in_loads}")

final_data = brt_dist.size_known_group_sparse_all_to_all(out_data, out_loads, in_loads)
# print(f"final tensor: {final_data}")
assert torch.allclose(tensor, final_data)
