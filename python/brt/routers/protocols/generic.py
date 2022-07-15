# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List

import torch
from brt.common import log
from brt.routers.utils import generate_dst_indices, generate_src_indices
from brt.routers.protocols.protocol import ProtocolBase, register_protocol

logger = log.get_logger(__file__)
__all__ = ["TopKProtocol", "ThresholdProtocol"]


@register_protocol("topk")
class TopKProtocol(ProtocolBase):
    def __init__(self, index_format="src_index", index_gen_opt=True, **kwargs):

        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.supported_capacities = kwargs.get("supported_capacities")
        self.k = kwargs.get("k")
        if self.k == None:
            self.k = 1
            logger.warning("k is not specified for Top-K protocol, use default k=1")

    def make_route_decision(self, score):
        hot_mask = torch.topk(score, self.k, dim=1).indices  # sample x k
        hot_mask = torch.zeros_like(
            score, dtype=torch.int64, device=score.device
        ).scatter_(1, hot_mask, 1)

        if self.index_format == "src_index":
            route_indices, loads = generate_src_indices(
                hot_mask, self.supported_capacities, self.index_gen_opt
            )
        elif self.index_format == "dst_index":
            route_indices, loads = generate_dst_indices(
                hot_mask, self.supported_capacities, self.index_gen_opt
            )

        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities


@register_protocol("threshold")
class ThresholdProtocol(ProtocolBase):
    def __init__(self, index_format="src_index", index_gen_opt=True, **kwargs):
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.supported_capacities = kwargs.get("supported_capacities")
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

    def make_route_decision(self, score):
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

        if self.index_format == "src_index":
            route_indices, loads = generate_src_indices(
                hot_mask, self.supported_capacities, self.index_gen_opt
            )
        elif self.index_format == "dst_index":
            route_indices, loads = generate_dst_indices(
                hot_mask, self.supported_capacities, self.index_gen_opt
            )

        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities
