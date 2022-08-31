# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import os
from typing import Callable, Dict, Type, Any

import torch
import torch.nn as nn
from brt.runtime import log
from brt.runtime import Registry
from brt.router.utils import make_kwargs

logger = log.get_logger(__file__)

__all__ = ["ProtocolBase", "ProtocolFactory"]


class ProtocolBase(nn.Module):
    def __init__(
        self,
        index_format: str = None,
        index_gen_opt: bool = True,
        **kwargs,
    ):
        """Base class for all protocols.

        Args:
            index_format (str): format of indices. should be "dst_index" or "src_index"
                src_index: the index of source for collecting data from input tensor, the index number start from zero
                dst_index: the index of destination for collected data from input tensor, the index number start from one,
                           zero is reserved for representing the corresponding data in the input tensor is dropped.
        """
        super().__init__()

        self.index_format = index_format
        self.index_gen_opt = index_gen_opt
        self.debug = os.environ.get("BRT_DEBUG", "False").lower() in ["true", "1"]
        assert self.index_format in [
            "dst_index",
            "src_index",
            None,
        ], f"index_format should be dst_index or src_index, but got {index_format}"

    def forward(self, score: torch.Tensor, **kwargs):
        decisions = self.make_route_decision(score, **kwargs)
        if self.debug:
            self.check_decision(decisions, score)
        return decisions

    def make_route_decision(self, score, **kwargs):
        raise NotImplementedError("make_route_decision has to be implemented by user")

    def check_decision(self, decisions, scores: torch.Tensor) -> bool:
        route_indices = decisions[0]
        loads = decisions[1]
        capacities = decisions[2]
        assert isinstance(route_indices, torch.Tensor) and isinstance(
            loads, torch.Tensor
        ), "indices and loads should be torch.Tensor"
        assert route_indices.size(0) == scores.size(
            0
        ), "indices and scores should have the same size in the first dimension"
        assert loads.numel() == route_indices.size(
            0
        ), "loads should have the same elements as indices in the first dimension"
        if isinstance(capacities, int):
            assert capacities >= 0, "capacities should be non-negative"
        elif isinstance(capacities, torch.Tensor):
            assert (
                capacities.size() == loads.size()
            ), "capacities should have the same elements as loads"
        else:
            raise ValueError("capacities should be int or torch.Tensor")

    def update(self, **kwargs):
        """Update the protocol.
        Some protocols may need to update their internal states.
        """
        raise NotImplementedError("Update has to be implemented by user")

    def check_compatibility(self, kwargs: Dict[str, Any]) -> None:
        pass


def register_protocol(protocol_type: str) -> Callable:
    return Registry.register_sub_cls(protocol_type, ProtocolBase)


def make_protocol(protocol_type: str, kwargs: Dict[str, Any]) -> ProtocolBase:
    protocol_cls = Registry.get_sub_cls(protocol_type, ProtocolBase)
    formulated_kwargs = make_kwargs(kwargs)
    return protocol_cls(**formulated_kwargs)
