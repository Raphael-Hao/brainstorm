# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.passes.base import PassBase, register_pass
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

@register_pass("pipline")
class PipelinePass(PassBase):
    pass

@register_pass("sharded")
class ShardedPass(PassBase):
    pass
