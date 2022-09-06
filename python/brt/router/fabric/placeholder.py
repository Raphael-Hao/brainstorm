# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Any, Dict
import torch
from brt.router.utils import assert_compatibility
from brt.router.fabric.base import register_fabric
from brt.router.fabric.generic import CombineFabric
from brt.runtime.proto_tensor import init_proto_tensor


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
        self.check_compatibility(kwargs)
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

    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        sparse = kwargs.pop("sparse", True)
        assert_compatibility(self, "sparse", True, sparse)

    def forward(self, in_flows: List[torch.Tensor]) -> torch.Tensor:
        if self.runtime_load == 1:
            if self.flow_num == 1:
                out_flows = torch.zeros(
                    self.ptu_grains[0],
                    dtype=self.ptu_dtypes[0],
                    device=self.ptu_devices[0],
                )
            else:
                out_flows = [
                    torch.zeros(
                        self.ptu_grains[i],
                        dtype=self.ptu_dtypes[i],
                        device=self.ptu_devices[i],
                    )
                    for i in range(self.flow_num)
                ]
        else:
            if self.flow_num == 1:
                out_flows = init_proto_tensor(
                    torch.zeros(
                        self.ptu_grains[0],
                        dtype=self.ptu_dtypes[0],
                        device=self.ptu_devices[0],
                    ),
                    [torch.zeros(0, 1, dtype=torch.int64, device=self.ptu_devices[0])],
                    [self.runtime_load],
                )
            else:
                out_flows = [
                    init_proto_tensor(
                        torch.zeros(
                            self.ptu_grains[i],
                            dtype=self.ptu_dtypes[i],
                            device=self.ptu_devices[i],
                        ),
                        [
                            torch.zeros(
                                0, 1, dtype=torch.int64, device=self.ptu_devices[i]
                            )
                        ],
                        [self.runtime_load],
                    )
                    for i in range(self.flow_num)
                ]
        return out_flows