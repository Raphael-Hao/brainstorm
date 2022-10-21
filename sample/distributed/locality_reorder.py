# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import torch
from brt._C import distributed
from brt.runtime.benchmark import CUDATimer

loads = torch.randint(1, 10, (10, 10), dtype=torch.int32).cuda()
world_size = 10

timer = CUDATimer(warm_up=100, loop=1000, repeat=10)
timer.execute(
    lambda: distributed.locality_reorder(loads, world_size), msg="locality_reorder"
)

# %%
