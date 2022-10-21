# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist
import brt.runtime.distributed as brt_dist

dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
device = torch.device('cuda', local_rank)
torch.cuda.set_device(device)
print(local_rank)
group = dist.group.WORLD
brt_dist.init_nccl(group)

