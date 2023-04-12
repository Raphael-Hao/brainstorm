# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Union, Tuple

import torch
from brt.router.fabric.base import register_fabric
from brt.router.fabric.generic import CombineFabric
from brt.runtime.grid_tensor import GridTensor, init_grid_tensor


@register_fabric("placeholder_combine")
class PlaceholderCombineFabric(CombineFabric):
    def __init__(
        self,
        flow_num: int,
        runtime_load: int,
        ptu_grains: List[torch.Size],
        ptu_dtypes: List[torch.dtype],
        ptu_devices: List[torch.device],
        **kwargs,
    ):
        super().__init__(flow_num=flow_num, sparse=True, **kwargs)
        self.runtime_load = runtime_load
        for i in range(self.flow_num):
            ptu_grains[i] = list(ptu_grains[i])
            ptu_grains[i][0] = 0
        self.ptu_grains = ptu_grains
        self.ptu_dtypes = ptu_dtypes
        self.ptu_devices = ptu_devices
        assert (
            len(self.ptu_grains) == self.flow_num
            and len(self.ptu_dtypes) == self.flow_num
            and len(self.ptu_devices) == self.flow_num
        ), "ptu_grains/dtypes/devices must have the same elements as flow_num"

    def forward(
        self,
        in_flow: Union[GridTensor, List[GridTensor]],
        hot_mask: torch.Tensor,
        runtime_capacities: torch.Tensor = None,
        score: torch.Tensor = None,
    ) -> Tuple[Union[List[GridTensor], List[List[GridTensor]]], torch.Tensor]:
        if self.flow_num == 1:
            out_flows = init_grid_tensor(
                torch.zeros(
                    self.ptu_grains[0],
                    dtype=self.ptu_dtypes[0],
                    device=self.ptu_devices[0],
                ),
                [torch.zeros(0, dtype=torch.int32, device=self.ptu_devices[0])],
                [torch.zeros(1, dtype=torch.int32, device=self.ptu_devices[0])],
            )
        else:
            out_flows = [
                init_grid_tensor(
                    torch.zeros(
                        self.ptu_grains[i],
                        dtype=self.ptu_dtypes[i],
                        device=self.ptu_devices[i],
                    ),
                    [torch.zeros(0, dtype=torch.int32, device=self.ptu_devices[i])],
                    [torch.zeros(1, dtype=torch.int32, device=self.ptu_devices[i])],
                )
                for i in range(self.flow_num)
            ]
        return out_flows, score
