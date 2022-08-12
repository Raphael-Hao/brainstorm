# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.router.utils import generate_indices
from brt.router.protocol.base import ProtocolBase, register_protocol

__all__ = ["TopKProtocol"]

logger = log.get_logger(__file__)


@register_protocol("topk")
class TopKProtocol(ProtocolBase):
    def __init__(
        self,
        top_k=1,
        supported_capacities: torch.Tensor = None,
        index_format="src_index",
        index_gen_opt=True,
    ):
        """Top-K protocol

        Args:
            top_k (int, optional): k for top selecting. Defaults to 1.
            supported_capacities (optional): _description_. Defaults to None.
            index_format (str, optional): index tensors according to destination or source. Defaults to "src_index".
            index_gen_opt (bool, optional): whether use optimized GPU kernel. Defaults to True.
        """
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.supported_capacities = supported_capacities
        self.top_k = top_k

    def make_route_decision(self, score):
        hot_mask = torch.topk(score, self.top_k, dim=1).indices  # sample x k
        hot_mask = torch.zeros_like(
            score, dtype=torch.int64, device=score.device
        ).scatter_(1, hot_mask, 1)
        route_indices, loads = generate_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )
        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities