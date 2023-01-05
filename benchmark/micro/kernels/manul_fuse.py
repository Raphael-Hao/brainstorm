# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import os

os.environ[
    "BRT_CACHE_PATH"
] = "/home/v-weihaocui/brainstorm_project/brainstorm_raphael/.cache"

import torch
import torch.nn as nn
from brt.jit import make_jit_kernel
from brt.runtime.benchmark import CUDATimer

linears = nn.ModuleList(nn.Linear(768, 3072, bias=False).cuda() for i in range(2))


in_data_1 = torch.ones(512, 768).cuda()
in_data_2 = torch.ones(128, 768).cuda()
weight = torch.ones(3072, 768).cuda()
manu_out = torch.zeros(512, 3072).cuda()
tvm_out = torch.zeros(512, 3072).cuda()

fused = make_jit_kernel(
    linears, sample_inputs=[in_data_1, in_data_2], opt_level="homo_fuse", rank=[11, 1]
)

in_data = torch.ones(512+128, 768).cuda()
out_data= torch.zeros(512+128, 3072).cuda()

fused(
    [in_data, out_data],
    [linears[0].weight, linears[1].weight],
    torch.tensor([512, 128], dtype=torch.int32),
)

print(out_data)
# %%

print()