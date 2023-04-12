# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.router.protocol.base import ProtocolBase, register_protocol

__all__ = ["TopKProtocol"]

logger = log.get_logger(__file__)


@register_protocol("topk")
class TopKProtocol(ProtocolBase):
    def __init__(
        self, top_k=1,
    ):
        """Top-K protocol
        Args:
            top_k (int, optional): k for top selecting. Defaults to 1.
            supported_capacities (optional): _description_. Defaults to None.
        """
        super().__init__()
        self.top_k = top_k

    def make_route_decision(self, score):
        hot_mask = torch.topk(score, self.top_k, dim=1).indices  # sample x k
        hot_mask = torch.zeros_like(
            score, dtype=torch.int32, device=score.device
        ).scatter_(1, hot_mask, 1)
        return hot_mask
