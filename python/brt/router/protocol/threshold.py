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
        single_tpu=False,
        supported_capacities=None,
        index_format="src_index",
        index_gen_opt=True,
    ):
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.threshold = threshold
        self.residual_path = residual_path
        self.single_tpu = single_tpu
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
        if self.single_tpu:
            if hot_mask.numel() > 0:
                loads = hot_mask.view(-1).cpu()
            else:
                loads = torch.zeros(hot_mask.size(1), dtype=torch.int32, device="cpu")
            return hot_mask, loads, loads
        route_indices, loads = generate_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )

        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities


@register_protocol("residual_threshold")
class ResidualThresholdProtocol(ProtocolBase):
    def __init__(
        self,
        threshold=0.0,
        residual_path=0,
        single_tpu=False,
        supported_capacities=None,
        index_format="src_index",
        index_gen_opt=True,
    ):
        if index_format != "src_index":
            logger.error("BatchedThresholdProtocol only supports src_index format")
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.threshold = threshold
        self.residual_path = residual_path
        self.single_tpu = single_tpu
        self.supported_capacities = supported_capacities

    def make_route_decision(self, score: torch.Tensor):

        hot_mask = (score.sum(dim=1, keepdim=True) < self.threshold).long()

        if self.residual_path == 0:
            hot_mask = torch.ones(
                score.size(0), 2, dtype=torch.int64, device=score.device
            ).scatter_(1, hot_mask, 0)
        elif self.residual_path == 1:
            hot_mask = torch.zeros(
                score.size(0), 2, dtype=torch.int64, device=score.device
            ).scatter_(1, hot_mask, 1)
        else:
            raise ValueError("drop_path should be 0 or 1")
        if self.single_tpu:
            if hot_mask.numel() > 0:
                loads = hot_mask.view(-1).cpu()
            else:
                loads = torch.zeros(hot_mask.size(1), dtype=torch.int32, device="cpu")
            return hot_mask, loads, loads
        route_indices, loads = generate_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )
        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities


@register_protocol("binary_threshold")
class BinaryThresholdProtocol(ProtocolBase):
    def __init__(
        self,
        threshold=0.0,
        selected_path=0,
        single_tpu=False,
        supported_capacities=None,
        index_format="src_index",
        index_gen_opt=True,
    ):
        if index_format != "src_index":
            logger.error("BatchedThresholdProtocol only supports src_index format")
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.threshold = threshold
        self.selected_path = selected_path
        self.single_tpu = single_tpu
        self.supported_capacities = supported_capacities

    def make_route_decision(self, score: torch.Tensor):

        hot_mask = (score > self.threshold).long()

        if self.selected_path == 0:
            hot_mask = torch.ones(
                score.size(0), 2, dtype=torch.int64, device=score.device
            ).scatter_(1, hot_mask, 0)
        elif self.selected_path == 1:
            hot_mask = torch.zeros(
                score.size(0), 2, dtype=torch.int64, device=score.device
            ).scatter_(1, hot_mask, 1)
        else:
            raise ValueError(
                "selected_path should be 0 or 1 for a binary threshold protocol"
            )

        if self.single_tpu:
            if hot_mask.numel() > 0:
                loads = hot_mask.view(-1).cpu()
            else:
                loads = torch.zeros(hot_mask.size(1), dtype=torch.int32, device="cpu")
            return hot_mask, loads, loads
        route_indices, loads = generate_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )
        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities
