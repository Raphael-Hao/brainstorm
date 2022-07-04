# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.frontend import nn


class Protocol:
    def __init__(self, path_num: int):
        self.path_num = path_num

    def gen_hot_mask(self, x):
        raise NotImplementedError()


class TopKProtocol(Protocol):
    def __init__(self, path_num: int, k: int = 1):
        super().__init__(path_num)
        self.k = k

    def gen_hot_mask(self, score):
        hot_mask = torch.topk(score, self.k, dim=1).indices  # sample x k
        hot_mask = torch.zeros(
            score.size(0), self.path_num, dtype=torch.int64, device=score.device
        ).scatter_(
            1, hot_mask, 1
        )  # sample x dst_num
        return hot_mask


class ThresholdProtocol(Protocol):
    def __init__(self, path_num: int, threshold: float):
        super().__init__(path_num)
        self.threshold = threshold

    def gen_hot_mask(self, score):
        hot_mask = (score > self.threshold).long().to(score.device)
        return hot_mask
