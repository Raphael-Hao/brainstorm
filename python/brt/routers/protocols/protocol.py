# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import os
from typing import Callable, Dict, Type

import torch
import torch.nn as nn
from brt.common import log
from brt.runtime import Registry

logger = log.get_logger(__file__)

__all__ = ["ProtocolBase", "ProtocolFactory"]


class ProtocolBase(nn.Module):
    def __init__(
        self,
        path_num: int,
        index_format: str,
        indices_gen_opt: bool = True,
    ):
        """Base class for all protocols.

        Args:
            path_num (int): number of paths for routing source or destinations
            index_format (str): format of indices. should be "dst_index" or "src_index"
                src_index: the index of source for collecting data from input tensor, the index number start from zero
                dst_index: the index of destination for collected data from input tensor, the index number start from one,
                           zero is reserved for representing the corresponding data in the input tensor is dropped.
        """
        super().__init__()
        self.path_num = path_num
        self.index_format = index_format
        self.indices_gen_opt = indices_gen_opt
        self.debug = os.environ.get("BRT_DEBUG", "False").lower() in ["true", "1"]
        assert self.path_num >= 1, f"path_num should be at least 1, but got {path_num}"
        assert self.index_format in [
            "dst_index",
            "src_index",
        ], f"index_format should be dst_index or src_index, but got {index_format}"

    def forward(self, score: torch.Tensor, **kwargs):
        decisions = self.make_route_decision(score, **kwargs)
        if self.debug:
            self.check_decision(decisions, score)
        return decisions

    def make_route_decision(self, score):
        raise NotImplementedError("make_route_decision has to be implemented by user")

    def check_decision(self, decisions, scores: torch.Tensor) -> bool:
        indices = decisions[0]
        loads = decisions[1]
        capacities = decisions[2]
        assert indices.size(0) == scores.size(
            0
        ), "indices and scores should have the same size in the first dimension"
        assert (
            loads.numel() == self.path_num
        ), "loads should have the same elements as path_num"
        if isinstance(capacities, int):
            assert capacities >= 0, "capacities should be non-negative"
        elif isinstance(capacities, torch.Tensor):
            assert (
                capacities.numel() == self.path_num
            ), "capacities should have the same elements as path_num"
        else:
            raise ValueError("capacities should be int or torch.Tensor")

    def update(self, **kwargs):
        """Update the protocol.
        Some protocols may need to update their internal states.
        """
        raise NotImplementedError("Update has to be implemented by user")


def register_protocol(protocol_type: str) -> Callable:
    return Registry.register_cls(protocol_type, ProtocolBase)


def make_protocol(protocol_type: str, **kwargs) -> ProtocolBase:
    protocol_cls = Registry.get_cls(protocol_type, ProtocolBase)
    return protocol_cls(**kwargs)
