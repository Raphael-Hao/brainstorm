# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict

import torch
import torch.fx
import torch.nn as nn
from brt.router.scatter import ScatterRouter

__all__ = ["RandScatter", "UniformScatter"]


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
        )

    def forward(self, inputs):
        score = rand_gate(inputs, self.path_num)
        route_results = self.scatter_router(inputs, score)
        return route_results


@torch.fx.wrap
def uniform_gate(x: torch.Tensor, path_num: int):
    assert x.size(0) % path_num == 0
    mask_indice = torch.arange(path_num, device=x.device).repeat_interleave(
        x.size(0) // path_num
    )
    mask = torch.zeros((x.size(0), path_num), device=x.device).scatter_(
        1, mask_indice.unsqueeze(1), 1
    )
    # print(mask)
    return mask


class UniformScatter(nn.Module):
    def __init__(
        self,
        path_num: int,
        fabric_type: str = "dispatch",
        protocol_kwargs: Dict[str, Any] = None,
        fabric_kwargs: Dict[str, Any] = None,
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
        )

    def forward(self, inputs):
        score = uniform_gate(inputs, self.path_num)
        route_results = self.scatter_router(inputs, score)
        return route_results


@torch.fx.wrap
def miss_hit_gate(x: torch.Tensor, path_num: int, is_hit: bool):
    if is_hit:
        mask_indice = torch.zeros((x.size(0), 1), device=x.device, dtype=torch.int64)
    else:
        # mask_indice = torch.randint(
        #     1, path_num, (x.size(0), 1), device=x.device, dtype=torch.int64
        # )
        mask_indice = torch.zeros(
            (x.size(0), 1), device=x.device, dtype=torch.int64
        ).fill_(path_num - 1)
    mask = torch.zeros((x.size(0), path_num), device=x.device).scatter_(
        1, mask_indice, 1
    )
    return mask


class MissHitScatter(nn.Module):
    def __init__(
        self,
        path_num: int,
        is_hit: bool,
        fabric_type: str = "dispatch",
        protocol_kwargs: Dict[str, Any] = None,
        fabric_kwargs: Dict[str, Any] = None,
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
        self.is_hit = is_hit
        self.scatter_router = ScatterRouter(
            protocol_type="topk",
            fabric_type=fabric_type,
            protocol_kwargs=protocol_kwargs,
            fabric_kwargs=fabric_kwargs,
        )

    def forward(self, inputs):
        score = miss_hit_gate(inputs, self.path_num, self.is_hit)
        route_results = self.scatter_router(inputs, score)
        return route_results
