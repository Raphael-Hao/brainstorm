# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, Type

import torch
import torch.nn as nn
from brt.common import log
from brt.runtime import Registry

logger = log.get_logger(__file__)

__all__ = ["ProtocolBase", "ProtocolFactory"]


class ProtocolBase(nn.Module):
    def __init__(self, path_num: int, indices_format: str):
        super().__init__()
        self.path_num = path_num
        self.indices_format = indices_format

    def forward(self, score: torch.Tensor):
        decisions = self.make_route_decision(score)
        # self.check_decision(decisions, score)
        return decisions

    def make_route_decision(self, score):
        raise NotImplementedError("make_route_decision has to be implemented by user")

    def check_decision(self, decisions, scores: torch.Tensor) -> bool:
        indices = decisions[0]
        loads = decisions[1]
        capacities = decisions[2]
        assert (
            indices.size() == scores.size()
        ), "indices and scores should have the same size"
        assert loads.numel() == scores.size(
            1
        ), "loads should have the same elements as scores in the second dimension"
        if isinstance(capacities, int):
            assert capacities >= 0, "capacities should be non-negative"
        elif isinstance(capacities, torch.Tensor):
            assert capacities.numel() == scores.size(
                1
            ), "capacities should have the same elements as scores in the second dimension"
        else:
            raise ValueError("capacities should be int or torch.Tensor")


def register_protocol(protocol_type: str) -> Callable:
    return Registry.register_cls(protocol_type, ProtocolBase)


def make_protocol(protocol_type: str, **kwargs) -> ProtocolBase:
    protocol_cls = Registry.get_cls(protocol_type, ProtocolBase)
    return protocol_cls(**kwargs)
