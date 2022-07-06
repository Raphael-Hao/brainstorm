# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable, Dict, Type

import torch
import torch.nn as nn
from brt.common import log

logger = log.get_logger(__file__)

__all__ = ["ProtocolBase", "ProtocolFactory", "TopKProtocol", "ThresholdProtocol"]


class ProtocolBase(nn.Module):
    def __init__(self, path_num: int):
        super().__init__()
        self.path_num = path_num

    def gen_aux(self, x):
        raise NotImplementedError()

    def gen_hot_mask(self, x):
        raise NotImplementedError()


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


@ProtocolFactory.register("topk")
class TopKProtocol(ProtocolBase):
    def __init__(self, path_num, **kwargs):
        super().__init__(path_num)
        self.k = kwargs.get("k")
        self.residual_path = kwargs.get("residual_path")
        if self.k == None:
            self.k = 1
            logger.warning("k is not specified for Top-K route method, use default k=1")
        if self.residual_path == None:
            self.residual_path = -1
            logger.warning(
                "residual_path is not specified for Threshold route method, use default residual_path=-1"
            )

    def forward(self, score):
        return self.gen_hot_mask(score)

    def gen_hot_mask(self, score):
        hot_mask = torch.topk(score, self.k, dim=1).indices  # sample x k
        hot_mask = torch.zeros(
            score.size(0), self.path_num, dtype=torch.int64, device=score.device
        ).scatter_(
            1, hot_mask, 1
        )  # sample x dst_num
        return hot_mask


@ProtocolFactory.register("threshold")
class ThresholdProtocol(ProtocolBase):
    def __init__(self, path_num, **kwargs):
        super().__init__(path_num)
        self.threshold = kwargs.get("threshold")
        self.residual_path = kwargs.get("residual_path")
        if self.threshold == None:
            self.threshold = 0.0
            logger.warning(
                "threshold is not specified for Threshold route method, use default threshold=0.0"
            )
        if self.residual_path == None:
            self.residual_path = -1
            logger.warning(
                "residual_path is not specified for Threshold route method, use default residual_path=-1"
            )

    def forward(self, score):
        return self.gen_hot_mask(score)

    def gen_hot_mask(self, score):
        if score.is_cuda:
            hot_mask = (score > self.threshold).long().to(score.device)
        else:
            hot_mask = (score > self.threshold).long().to(score.device)

        if self.residual_path >= 0:
            residual_indices = (
                (hot_mask.sum(dim=1, keepdim=True) == 0).long().to(score.device)
            )  # [bs x 1]
            residual_index = torch.full(
                (residual_indices.shape),
                self.residual_path,
                dtype=torch.int64,
                device=score.device,
            )
            hot_mask = torch.scatter_add(hot_mask, 1, residual_index, residual_indices)

        return hot_mask
