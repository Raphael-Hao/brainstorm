# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List

import torch
from brt._C.router import generate_dst_indices
from brt.common import log
from brt.routers.protocols.protocol import ProtocolBase, register_protocol

logger = log.get_logger(__file__)
__all__ = ["TopKProtocol", "ThresholdProtocol"]


@register_protocol("topk")
class TopKProtocol(ProtocolBase):
    def __init__(self, path_num, **kwargs):
        super().__init__(path_num, indices_format="src_index")
        self.k = kwargs.get("k")
        if self.k == None:
            self.k = 1
            logger.warning("k is not specified for Top-K protocol, use default k=1")

    def make_route_decision(self, score):
        hot_mask = torch.topk(score, self.k, dim=1).indices  # sample x k
        hot_mask = torch.zeros(
            score.size(0), self.path_num, dtype=torch.int64, device=score.device
        ).scatter_(
            1, hot_mask, 1
        )  # sample x dst_num
        return self.gen_indices(hot_mask)

    def gen_indices(self, hot_mask: torch.Tensor) -> List[torch.Tensor]:
        """generate indices according to hot_mask

        Args:
            hot_mask (torch.Tensor): a multi-hot mask for representing the routing destinations of each sample
        """
        if hot_mask.is_cuda and self.indices_gen_opt:
            route_indices, loads = generate_dst_indices(hot_mask)
        else:
            route_indices = torch.zeros_like(hot_mask)
            loads = torch.zeros(
                (self.path_num,), dtype=torch.int64, device=hot_mask.device
            )
            hot_mask_t = hot_mask.t().contiguous()
            for i in range(self.path_num):
                src_indices = hot_mask_t[i].view(-1).nonzero()
                loads[i] = src_indices.numel()
                route_indices[i, : src_indices.numel()] = src_indices
        return route_indices


@register_protocol("threshold")
class ThresholdProtocol(ProtocolBase):
    def __init__(self, path_num, **kwargs):
        super().__init__(path_num, indices_format="src_index")
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
        return self.gen_indices(hot_mask)

    def gen_indices(self, hot_mask: torch.Tensor) -> List[torch.Tensor]:
        """generate indices according to hot_mask

        Args:
            hot_mask (torch.Tensor): a multi-hot mask for representing the routing destinations of each sample
        """
        if hot_mask.is_cuda and self.indices_gen_opt:
            route_indices = generate_dst_indices(hot_mask)
        else:
            route_indices = []
            hot_mask_t = hot_mask.t().contiguous()
            for i in range(self.path_num):
                route_indices.append(torch.nonzero(hot_mask_t[i].view(-1)))
        return route_indices
