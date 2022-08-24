# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Dict, Any
import torch
import torch.fx
import torch.nn as nn
from brt.router.scatter import ScatterRouter

__all__ = ["RandScatter"]


@torch.fx.wrap
def rand_gate(x: torch.Tensor, path_num: int):
    return torch.randn((x.size(0), path_num), device=x.device)


class RandScatter(nn.Module):
    def __init__(
        self,
        path_num: int,
        fabric_type: str = "dispatch",
        protocol_kwargs: Dict[str, Any] = None,
        fabric_kwargs: Dict[str, Any] = None,
        capturing=False,
        capture_mode: str = "c",
    ):
        """random scatter router

        Args:
            rand_path_num (int): number of paths for random routing destinations.
            fabric_type (str, optional): fabric type. Defaults to "dispatch".
                dispatch: dispatch the results to each path, independently.
                homo_fused_dispatch: dispatch the results to all paths in the form of a fused tensor
                    supported_capacities: (List[int]): required for homo_fused_dispatch
        """

        super().__init__()
        self.path_num = path_num
        self.scatter_router = ScatterRouter(
            protocol_type="topk",
            fabric_type=fabric_type,
            protocol_kwargs=protocol_kwargs,
            fabric_kwargs=fabric_kwargs,
            capturing=capturing,
            capture_mode=capture_mode,
        )

    def forward(self, inputs):
        score = rand_gate(inputs, self.path_num)
        route_results = self.scatter_router(inputs, score)
        return route_results
