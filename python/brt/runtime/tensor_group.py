# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Dict, Tuple

import torch
import torch.nn as nn


__all__ = ["group_params_buffers"]


class TensorGroup:
    pass

def group_params_buffers(
    params: Dict[str, nn.Parameter],
    buffers: Dict[str, Tuple[torch.Tensor, nn.Module, str]],
    target_device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group params and buffers into a single tensor.

    Args:
        params (Dict[str, nn.Parameter]): collected params
        buffers (Dict[str, nn.Parameter]): collected buffers

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: single pin memory and cuda tensor containing all params and buffers


    """
    if target_device == None:
        target_device = "cuda"
    t_size = 0
    for p in params.values():
        t_size += p.numel() * p.element_size()
    for b, _, _ in buffers.values():
        t_size += b.numel() * b.element_size()
    t_pin = torch.empty(t_size, dtype=torch.uint8).pin_memory()

    # TODO: support sharing the storage of params and buffers
    # Currently, the genereated single tensor does not share the storage
    # with params and buffers. In the future, we will remove this limitation.
    # See [https://github.com/pytorch/pytorch/blob/master/torch/multiprocessing/reductions.py]
    # and [https://github.com/pytorch/pytorch/blob/1378561d03d5bb1433f6404e829b49caaaba9e00/torch/_utils.py#L144]

    t_target = t_pin.to(target_device)

    return t_pin, t_target
