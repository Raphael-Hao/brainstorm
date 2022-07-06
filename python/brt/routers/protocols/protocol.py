# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, Type

import torch
import torch.nn as nn
from brt.common import log

logger = log.get_logger(__file__)

__all__ = ["ProtocolBase", "ProtocolFactory"]


class ProtocolBase(nn.Module):
    def __init__(self, path_num: int):
        super().__init__()
        self.path_num = path_num
    
    @torch.jit.ignore
    def forward(self, score: torch.Tensor):
        decisions = self.make_route_decision(score)
        self.check_decision(decisions, score)
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


class ProtocolFactory:
    registry: Dict[str, ProtocolBase] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def register_func(protocol_cls) -> Type[ProtocolBase]:
            if name in cls.registry:
                logger.warning(f"Protocol: {name} is already registered, overwrite it")
            if not issubclass(protocol_cls, ProtocolBase):
                raise ValueError(f"Protocol: {name} is not a subclass of ProtocolBase")
            protocol_cls.forward = torch.jit.ignore(protocol_cls.forward)
            cls.registry[name] = protocol_cls
            return protocol_cls

        return register_func

    @classmethod
    def make_protocol(cls, protocol_type, **kwargs) -> ProtocolBase:
        for key, value in kwargs.items():
            logger.debug(f"{key}: {value}")

        if protocol_type not in cls.registry:
            raise ValueError(f"Protocol: {protocol_type} is not registered")
        protocol_cls = cls.registry[protocol_type]
        protocol_inst = protocol_cls(**kwargs)

        return protocol_inst
