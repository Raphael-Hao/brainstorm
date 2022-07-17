# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.fx
import torch.nn as nn
from brt.router.generic import ScatterRouter

__all__ = ["RandScatter"]


@torch.fx.wrap
def rand_gate(x: torch.Tensor, path_num: int):
    return torch.randn((x.size(0), path_num), device=x.device)


class RandScatter(nn.Module):
    def __init__(self, path_num: int, fabric_type: str = "dispatch", **kwargs):
        """random scatter router

        Args:
            path_num (int): number of paths for routing destinations
            fabric_type (str, optional): fabric type. Defaults to "dispatch".
                dispatch: dispatch the results to each path, independently.
                homo_fused_dispatch: dispatch the results to all paths in the form of a fused tensor
                    supported_capacities: (List[int]): required for homo_fused_dispatch
        """

        super().__init__()
        self.path_num = path_num
        self.scatter_router = ScatterRouter(
            protocol_type="topk", fabric_type=fabric_type, k=1, **kwargs
        )

    def forward(self, inputs):
        score = rand_gate(inputs, self.path_num)
        route_results = self.scatter_router(inputs, score)
        return route_results