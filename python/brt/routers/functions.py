# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Tuple

import brt._C as _C
import torch

__all__ = ["generate_src_indices", "generate_dst_indices"]

def generate_src_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor = None,
    indices_gen_opt=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """generate source indices according to hot_mask

    Args:
        hot_mask (torch.Tensor): hot mask for representing the routing decisions
        supported_capacities (torch.Tensor, optional): sorted supported capacities. Defaults to None.
        indices_gen_opt (bool, optional): if use brt optimized indices generation GPU kernels. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: src indices and loads
    """

    if hot_mask.is_cuda and indices_gen_opt:
        src_indices, loads = _C.generate_src_indices(hot_mask, supported_capacities)
    else:
        src_indices = torch.zeros_like(hot_mask)
        loads = torch.zeros(
            (hot_mask.size(1),), dtype=torch.int64, device=hot_mask.device
        )
        hot_mask_t = hot_mask.t().contiguous()
        for i in range(hot_mask.size(1)):
            src_indices_per_path = hot_mask_t[i].view(-1).nonzero()
            loads[i] = src_indices_per_path.numel()
            src_indices[i, : src_indices_per_path.numel()] = src_indices_per_path
            if supported_capacities is not None:
                for capacity in supported_capacities:
                    if loads[i] <= capacity:
                        loads[i] = capacity
                        break
    return src_indices, loads


def generate_dst_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor = None,
    indices_gen_opt=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """generate destination indices according to hot_mask

    Args:
        hot_mask (torch.Tensor): hot mask for representing the routing decisions
        supported_capacities (torch.Tensor, optional): sorted supported capacities. Defaults to None.
        indices_gen_opt (bool, optional): if use brt optimized indices generation GPU kernels. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: destination indices and loads
    """
    if hot_mask.is_cuda and indices_gen_opt:
        dst_indices, loads = _C.generate_dst_indices(hot_mask, supported_capacities)
    else:
        dst_indices = torch.cumsum(hot_mask, dim=0) * hot_mask
        loads = torch.sum(hot_mask, dim=0)
        if supported_capacities is not None:
            for i in range(hot_mask.size(1)):
                real_load = loads[i]
                for capacity in supported_capacities:
                    if real_load <= capacity:
                        loads[i] = real_load
                        break
    return dst_indices, loads
