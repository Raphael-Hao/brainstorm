# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List

import torch
import numpy as np

__all__ = [
    "pad_to_max",
    "make_kwargs",
    "empty_flows",
]


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
