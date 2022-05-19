# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# %%
import brt
import torch

from thor_config import ThorConfig
from thor_moe import FusedThorMoE

config = ThorConfig()
config.token_num = 64
config.hidden_size = 512
config.intermediate_size = 1024
config.num_attention_heads = 8
config.num_hidden_layers = 1
config.expert_num = 2

fused_thor_moe = brt.domain(FusedThorMoE)(config)

# %%

fused_thor_moe.cuda()

x = torch.zeros(1, 64, 512).cuda()
x = fused_thor_moe(x)

print(x)

# %%
x = torch.zeros(1, 64, 512).cuda()
torch.cuda.synchronize()
stream = torch.cuda.default_stream()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record(stream)
for i in range(10):
    y = fused_thor_moe(x)
end_event.record(stream)
stream.synchronize()
print("elapsed time: {:.3f}".format(start_event.elapsed_time(end_event)/10))

print(y)
# %%
