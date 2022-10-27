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
print(f"in_loads: {loads}")
all_in_loads = None
if local_rank == 0:
    all_in_loads = [torch.empty_like(loads) for _ in range(world_size)]

dist.gather(loads, all_in_loads)
if local_rank == 0:
    all_in_loads = torch.stack(all_in_loads)
    print(f"all_in_loads: {all_in_loads}")

out_data, out_loads, reorder_indices = brt_dist.asymmetry_a2a(
    tensor, loads, locality_aware=True
)
all_out_loads = None
if local_rank == 0:
    all_out_loads = [torch.empty_like(out_loads) for _ in range(world_size)]
dist.gather(out_loads, all_out_loads)
if local_rank == 0:
    all_out_loads = torch.stack(all_out_loads)
    print(f"all_out_loads: {all_out_loads}")
    print(reorder_indices)

timer = CUDATimer(repeat=2, root=local_rank)

timer.execute(
    lambda: brt_dist.asymmetry_a2a(tensor, loads),
    "brt.asymmetry_all_to_all",
)


# def locality_aware_a2a(tensor, loads):
#     out_data, out_loads, reorder_indices = brt_dist.asymmetry_a2a(
#         tensor, loads, locality_aware=True
#     )
#     print(reorder_indices)
#     out_data = out_data.reshape(world_size, -1, out_data.size(1))
#     print(out_data.shape)
#     out_data = out_data[reorder_indices.long()].reshape(-1, out_data.size(2))
#     final_out, _ = brt_dist.asymmetry_a2a(out_data, out_loads, locality_aware=False)
#     return final_out


# locality_aware_a2a(tensor, loads)

timer.execute(
    lambda: brt_dist.asymmetry_a2a(tensor, loads, locality_aware=True),
    "brt.asymmetry_all_to_all with locality aware",
)


def torch_symmetry_a2a(tensor, loads):
    out_loads = torch.empty_like(loads)
    dist.all_to_all_single(out_loads, loads)
    torch.cuda.synchronize()
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor)
    return output


timer.execute(lambda: torch_symmetry_a2a(tensor, loads), "dist.all_to_all_single")
