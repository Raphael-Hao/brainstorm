# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.router.protocol.base import ProtocolBase, register_protocol
import torch.nn as nn

__all__ = ["ThresholdProtocol"]

logger = log.get_logger(__file__)


@register_protocol("threshold")
class ThresholdProtocol(ProtocolBase):
    def __init__(
        self, threshold=0.0, residual_path=-1, **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.residual_path = residual_path

    def make_route_decision(self, score: torch.Tensor):

        hot_mask = (score > self.threshold).to(torch.int32, non_blocking=True)
        # print(f"===========prev residual===========\n{hot_mask}")
        if self.residual_path >= 0:
            residual_indices = (hot_mask.sum(dim=1, keepdim=True) == 0).to(torch.int32, non_blocking=True)
            # print(f"===========residual_indices===========\n{residual_indices}")
            residual_index = torch.full(
                (residual_indices.shape),
                self.residual_path,
                dtype=torch.int64,
                device=score.device,
            )
            hot_mask = torch.scatter_add(hot_mask, 1, residual_index, residual_indices)
        # print(f"===========post residual===========\n{hot_mask}")
        return hot_mask.to(torch.int32, non_blocking=True)


@register_protocol("residual_threshold")
class ResidualThresholdProtocol(ProtocolBase):
    def __init__(
        self, threshold=0.0, residual_path=0, **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.residual_path = residual_path

    def make_route_decision(self, score: torch.Tensor):

        indices = (score.sum(dim=1, keepdim=True) < self.threshold).to(torch.int64, non_blocking=True)
        if self.residual_path == 0:
            hot_mask = torch.ones(
                score.size(0), 2, dtype=torch.int32, device=score.device
            ).scatter_(1, indices, 0)
        elif self.residual_path == 1:
            hot_mask = torch.zeros(
                score.size(0), 2, dtype=torch.int32, device=score.device
            ).scatter_(1, indices, 1)
        else:
            raise ValueError("residual_path should be 0 or 1")

        return hot_mask


@register_protocol("binary_threshold")
class BinaryThresholdProtocol(ProtocolBase):
    def __init__(
        self, threshold=0.0, selected_path=0, **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.selected_path = selected_path
        self.softmax = nn.Softmax(dim=1)

    def make_route_decision(self, score: torch.Tensor):
        logit_score = self.softmax(score)
        max_preds, _argmax_preds = logit_score.max(dim=1, keepdim=False)
        hot_mask = max_preds >= self.threshold
        hot_mask = hot_mask.unsqueeze(-1)
        if self.selected_path == 0:
            hot_mask = torch.ones(
                score.size(0), 2, dtype=torch.int32, device=score.device
            ).scatter_(1, hot_mask, 0)
        elif self.selected_path == 1:
            hot_mask = torch.zeros(
                score.size(0), 2, dtype=torch.int32, device=score.device
            ).scatter_(1, hot_mask, 1)
        else:
            raise ValueError(
                "selected_path should be 0 or 1 for a binary threshold protocol"
            )
        return hot_mask
