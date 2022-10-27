# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist
from brt.router.fabric.base import register_fabric
from brt.router.fabric.generic import DispatchFabric, CombineFabric
from brt.router.fabric.fused import HomoFusedDispatchFabric, HomoFusedCombineFabric


@register_fabric("distributed_dispatch")
class DistributedDispatchFabric(DispatchFabric):
    def placement(self, src_index, dst_index):
        pass


@register_fabric("distributed_combine")
class CombineDispatchFabric(CombineFabric):
    pass
