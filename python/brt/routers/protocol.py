# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.common import log
from brt.frontend import nn

logger = log.get_logger(__file__)


class Protocol(nn.Module):
    def __init__(self, path_num: int):
        self.path_num = path_num

    def gen_aux(self, x):
        raise NotImplementedError()

    def gen_hot_mask(self, x):
        raise NotImplementedError()


class TopKProtocol(Protocol):
    def __init__(self, path_num: int, **kwargs):
        super().__init__(path_num)
        self.k = kwargs.get("k")
        if self.k == None:
            self.k = 1
            logger.warning("k is not specified for Top-K route method, use default k=1")
        if self.residual_dst == None:
            self.residual_dst = -1
            logger.warning(
                "residual_dst is not specified for Threshold route method, use default residual_dst=-1"
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


class ThresholdProtocol(Protocol):
    def __init__(self, path_num: int, **kwargs):
        super().__init__(path_num)
        self.threshold = kwargs.get("threshold")
        self.residual_dst = kwargs.get("residual_dst")
        if self.threshold == None:
            self.threshold = 0.0
            logger.warning(
                "threshold is not specified for Threshold route method, use default threshold=0.0"
            )
        if self.residual_dst == None:
            self.residual_dst = -1
            logger.warning(
                "residual_dst is not specified for Threshold route method, use default residual_dst=-1"
            )

    def forward(self, score):
        return self.gen_hot_mask(score)

    def gen_hot_mask(self, score):

        hot_mask = (score > self.threshold).long().to(score.device)

        if self.residual_dst >= 0:
            residual_indices = (
                (hot_mask.sum(dim=1, keepdim=True) == 0).long().to(score.device)
            )  # [bs x 1]
            residual_index = torch.full(
                (residual_indices.shape),
                self.residual_dst,
                dtype=torch.int64,
                device=score.device,
            )
            hot_mask = torch.scatter_add(hot_mask, 1, residual_index, residual_indices)

        return hot_mask
