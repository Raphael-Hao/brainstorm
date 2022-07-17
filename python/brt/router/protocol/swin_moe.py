# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import os
import torch
from brt.common import log
from brt.router.protocol.base import ProtocolBase, register_protocol

logger = log.get_logger(__file__)


@register_protocol("swin_moe")
class SwinMoEProtocol(ProtocolBase):
    def __init__(self, **kwargs):
        super().__init__()
        top_k = kwargs.get("top_k")
        capacity_factor = kwargs.get("capacity_factor")
        num_global_experts = kwargs.get("num_global_experts")
        self.top_k = min(top_k, num_global_experts)
        self.capacity_factor = float(os.environ.get("CAP_FACTOR", capacity_factor))
        self.normalize_gate = kwargs.get("normalize_gate")
        self.vitmoe_loss = kwargs.get("vitmoe_loss")
        self.use_noise = kwargs.get("use_noise")
        if self.vitmoe_loss:
            logger.warning(
                "change use_noise in TopKGate to True because vitmoe_loss is set to True"
            )
            self.use_noise = True
        self.batch_prioritized_routing = kwargs.get("batch_prioritized_routing")
        if int(os.environ.get("BATCH_PRIO", 0)) != 0:
            self.batch_prioritized_routing = True
        self.use_global_loss = kwargs.get("use_global_loss")
        self.is_postscore = kwargs.get("is_postscore")

    def forward(self, score, gates):
        hot_mask = self.gen_hot_mask(score)

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
