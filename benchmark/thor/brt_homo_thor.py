# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# %%
import torch

from .thor_config import ThorConfig
from .thor_model import ThorEncoder
from .thor_moe import FusedThorMoE # pylint: disable=unused-import

config = ThorConfig()
config.token_num = 64
config.hidden_size = 512
config.intermediate_size = 1024
config.num_attention_heads = 8
config.num_hidden_layers = 1
config.expert_num = 16
config.expert_type = "brt_homo_moe"

# fused_thor_moe = FusedThorMoE(config).eval()
fused_thor_moe = ThorEncoder(config).eval()


fused_thor_moe.cuda()

x = torch.randn(1, 4, 512).cuda()
x = fused_thor_moe(x)


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
print("elapsed time: {:.3f}".format(start_event.elapsed_time(end_event) / 10))
