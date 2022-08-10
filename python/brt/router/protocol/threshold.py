# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.router.utils import generate_indices
from brt.router.protocol.base import ProtocolBase, register_protocol

__all__ = ["ThresholdProtocol"]

logger = log.get_logger(__file__)


@register_protocol("threshold")
class ThresholdProtocol(ProtocolBase):
    def __init__(
        self,
        threshold=0.0,
        residual_path=-1,
        supported_capacities=None,
        index_format="src_index",
        index_gen_opt=True,
    ):
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.threshold = threshold
        self.residual_path = residual_path
        self.supported_capacities = supported_capacities

    def make_route_decision(self, score: torch.Tensor):
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

        route_indices, loads = generate_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )

        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities


@register_protocol("batched_threshold")
class ThresholdDropProtocol(ProtocolBase):
    def __init__(
        self,
        threshold=0.0,
        drop_path=1,
        index_format="src_index",
        index_gen_opt=True,
    ):
        if index_format != "src_index":
            logger.error("ThresholdDropProtocol only supports src_index format")
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.threshold = torch.tensor(threshold, dtype=torch.float)
        self.drop_path = drop_path

    def make_route_decision(self, score: torch.Tensor):

        self.threshold.to(score.device)
        hot_mask = (score.sum(dim=1, keepdim=True) < self.threshold).long()

        hot_mask = torch.zeros(
            score.size(0), 2, dtype=torch.int64, device=score.device
        ).scatter_(1, hot_mask, 1)

        route_indices, loads = generate_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )
        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities
