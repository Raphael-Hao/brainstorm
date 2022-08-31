# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Tuple, List

import torch
import numpy as np
import brt._C.router as C_router

__all__ = [
    "generate_src_indices",
    "generate_dst_indices",
    "generate_indices",
    "convert_index_format",
    "make_kwargs",
    "empty_flows",
]


def generate_src_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor = None,
    index_gen_opt=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """generate source indices according to hot_mask

    Args:
        hot_mask (torch.Tensor): hot mask for representing the routing decisions
        supported_capacities (torch.Tensor, optional): sorted supported capacities. Defaults to None.
        index_gen_opt (bool, optional): if use brt optimized indices generation GPU kernels. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: src indices and loads
    """

    if hot_mask.is_cuda and index_gen_opt:
        src_indices, loads = C_router.generate_src_indices(
            hot_mask.to(torch.int32), supported_capacities
        )
    else:
        src_indices = torch.zeros_like(hot_mask)
        # loads = [0 for _ in range(hot_mask.size(1))]
        loads = torch.zeros(hot_mask.size(1), dtype=torch.int32, device="cpu")
        torch.zeros((hot_mask.size(1),), dtype=torch.int64, device=hot_mask.device)
        hot_mask_t = hot_mask.t().contiguous()
        for i in range(hot_mask.size(1)):
            src_indices_per_path = hot_mask_t[i].view(-1).nonzero()
            loads[i] = src_indices_per_path.numel()
            src_indices[
                : src_indices_per_path.numel(), i : i + 1
            ] = src_indices_per_path
            if supported_capacities is not None:
                for capacity in supported_capacities:
                    if loads[i] <= capacity:
                        loads[i] = capacity
                        break
    return src_indices, loads


def generate_dst_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor = None,
    index_gen_opt=True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """generate destination indices according to hot_mask

    Args:
        hot_mask (torch.Tensor): hot mask for representing the routing decisions
        supported_capacities (torch.Tensor, optional): sorted supported capacities. Defaults to None.
        index_gen_opt (bool, optional): if use brt optimized indices generation GPU kernels. Defaults to True.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: destination indices and loads
    """
    if hot_mask.is_cuda and index_gen_opt:
        dst_indices, loads = C_router.generate_dst_indices(
            hot_mask.to(torch.int32), supported_capacities
        )
    else:
        dst_indices = torch.cumsum(hot_mask, dim=0) * hot_mask
        loads = torch.sum(hot_mask, dim=0).cpu()
        if supported_capacities is not None:
            for i in range(hot_mask.size(1)):
                real_load = loads[i]
                for capacity in supported_capacities:
                    if real_load <= capacity:
                        loads[i] = real_load
                        break
    return dst_indices, loads


def generate_indices(
    hot_mask: torch.Tensor,
    supported_capacities: torch.Tensor = None,
    index_format: str = "src_index",
    index_gen_opt=True,
):
    if index_format == "src_index":
        indices, loads = generate_src_indices(
            hot_mask, supported_capacities, index_gen_opt
        )
    elif index_format == "dst_index":
        indices, loads = generate_dst_indices(
            hot_mask, supported_capacities, index_gen_opt
        )
    else:
        raise ValueError(f"Unknown index format: {index_format}")
    return indices, loads


def convert_index_format(
    origin_indices: torch.Tensor,
    loads: torch.Tensor,
    origin_index_fmt: str,
    new_index_fmt: str,
):
    """
    Convert the route indices to the new index format.
    """
    if origin_index_fmt == new_index_fmt:
        return origin_indices
    elif new_index_fmt == "src_index":
        return C_router.convert_index_format(origin_indices, loads, 0)
    elif new_index_fmt == "dst_index":
        return C_router.convert_index_format(origin_indices, loads, 1)
    else:
        raise ValueError(f"Unknown index format: {new_index_fmt}")


def pad_to_max(tensors: List[torch.Tensor], pad_value=0):
    """
    Pad all the tensors to the max shape.
    """
    lengths = [t.numel() // t.size(0) for t in tensors]

    max_id = np.argmax(lengths)

    max_length = lengths[max_id]

    padded_shape = tensors[max_id].shape[1:]

    tensors = [t.view(t.size(0), -1) for t in tensors]

    pads = [(0, max_length - l) for l in lengths]

    tensors = [
        torch.nn.functional.pad(t, pad=p, mode="constant", value=pad_value).view(
            t.size(0), *padded_shape
        )
        for t, p in zip(tensors, pads)
    ]

    return tensors


def make_kwargs(kwargs):
    if kwargs is None:
        return {}
    if isinstance(kwargs, dict) and all(isinstance(k, str) for k in kwargs):
        return kwargs
    else:
        raise ValueError(
            "kwargs should be a dict of str to Any, but got {}".format(kwargs)
        )


def empty_flows(in_flows):
    if isinstance(in_flows, List):
        if len(in_flows) == 0:
            return True
        else:
            return empty_flows(in_flows[0])
    return False


def assert_compatibility(instance, k, expected_v, given_v) -> None:
    assert (
        expected_v == given_v
    ), f"compatibility check failed for {type(instance).__name__},\
                caused by keyword argument{k}: expected {expected_v}, given {given_v}"
