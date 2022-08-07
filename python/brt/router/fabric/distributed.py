# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist
from brt.router.fabric.generic import DispatchFabric, CombineFabric
from brt.router.fabric.homo_fused import HomoFusedDispatchFabric, HomoFusedCombineFabric


tensors = [torch.randn((4, 4)) for i in range(4)]
dist.all_to_all(tensors)
