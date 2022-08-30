# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List
import torch
from brt.router.fabric.base import register_fabric
from brt.router.fabric.generic import CombineFabric


@register_fabric("placeholder_combine")
class PlaceholderCombineFabric(CombineFabric):
    def __init__(
        self,
        flow_num: int,
        flow_shapes: List[torch.Size],
        flow_dtypes: List[torch.dtype],
        flow_devices: List[torch.device],
        **kwargs,
    ):
        super().__init__(flow_num=flow_num, sparse=True, **kwargs)
        for i in range(self.flow_num):
            flow_shapes[i][0] = 0
        self.flow_shapes = flow_shapes
        self.flow_dtypes = flow_dtypes
        self.flow_devices = flow_devices
        assert (
            len(self.flow_shapes) == self.flow_num
            and len(self.flow_dtypes) == self.flow_num
            and len(self.flow_devices) == self.flow_num
        ), "flow_shapes/dtypes/devices must have the same elements as flow_num"

    def forward(self, in_flows: List[torch.Tensor]) -> torch.Tensor:
        if self.flow_num == 1:
            return torch.zeros(
                self.flow_shapes[0],
                dtype=self.flow_dtypes[0],
                device=self.flow_devices[0],
            )
        return [
            torch.zeros(
                self.flow_shapes[i],
                dtype=self.flow_dtypes[i],
                device=self.flow_devices[i],
            )
            for i in range(self.flow_num)
        ]
