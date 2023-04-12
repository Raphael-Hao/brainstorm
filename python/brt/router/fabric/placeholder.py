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
        cell_grains: List[torch.Size],
        cell_dtypes: List[torch.dtype],
        cell_devices: List[torch.device],
        **kwargs,
    ):
        super().__init__(flow_num=flow_num, sparse=True, **kwargs)
        self.runtime_load = runtime_load
        for i in range(self.flow_num):
            cell_grains[i] = list(cell_grains[i])
            cell_grains[i][0] = 0
        self.cell_grains = cell_grains
        self.cell_dtypes = cell_dtypes
        self.cell_devices = cell_devices
        assert (
            len(self.cell_grains) == self.flow_num
            and len(self.cell_dtypes) == self.flow_num
            and len(self.cell_devices) == self.flow_num
        ), "cell_grains/dtypes/devices must have the same elements as flow_num"

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
                    self.cell_grains[0],
                    dtype=self.cell_dtypes[0],
                    device=self.cell_devices[0],
                ),
                [torch.zeros(0, dtype=torch.int32, device=self.cell_devices[0])],
                [torch.zeros(1, dtype=torch.int32, device=self.cell_devices[0])],
            )
        else:
            out_flows = [
                init_grid_tensor(
                    torch.zeros(
                        self.cell_grains[i],
                        dtype=self.cell_dtypes[i],
                        device=self.cell_devices[i],
                    ),
                    [torch.zeros(0, dtype=torch.int32, device=self.cell_devices[i])],
                    [torch.zeros(1, dtype=torch.int32, device=self.cell_devices[i])],
                )
                for i in range(self.flow_num)
            ]
        return out_flows, score
