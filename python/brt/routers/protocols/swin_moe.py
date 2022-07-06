# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.common import log
from brt.routers.protocols.protocol import ProtocolBase, ProtocolFactory

logger = log.get_logger(__file__)


class SwinMoEProtocol(ProtocolBase):
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

    def gen_aux(self, x):
        pass
