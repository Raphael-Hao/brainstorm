# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import torch
from brt._C import distributed as dist
from brt.runtime.benchmark import CUDATimer

world_size = 8
group_size = 4
capacity = 1024
group_loads = torch.randint(
    1, capacity, (world_size, world_size, group_size), dtype=torch.int32
).cuda()

# reordered_loads, reorder_indices = dist.locality_reorder(loads, world_size)
# print(reordered_loads)
# print(reorder_indices)
print(group_loads)
group_reordered_loads, group_reorder_indices = dist.group_locality_reorder(
    group_loads, world_size, group_size
)
print(group_reordered_loads)
print(group_reorder_indices)
timer = CUDATimer(warm_up=100, loop=1000, repeat=10)
timer.execute(
    lambda: dist.group_locality_reorder(group_loads, world_size, group_size),
    msg="group_locality_reorder",
)

timer.execute(
    lambda: dist.group_locality_reorder(group_loads, world_size, group_size=1),
    msg="locality_reorder",
)


# reordered_loads, reorder_indices = dist.locality_reorder(loads, world_size)
# print(reordered_loads)
# print(reorder_indices)

# timer.execute(lambda: dist.locality_reorder(loads, world_size), msg="locality_reorder")


# %%
