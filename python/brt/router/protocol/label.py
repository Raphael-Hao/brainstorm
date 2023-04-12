# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.router.protocol.base import ProtocolBase, register_protocol

__all__ = ["LabelProtocol"]

logger = log.get_logger(__file__)


@register_protocol("label")
class LabelProtocol(ProtocolBase):
    def __init__(
        self, path_num: int, **kwargs,
    ):
        """Label protocol

        Args:
            supported_capacities (optional): _description_. Defaults to None.
            index_format (str, optional): index tensors according to destination or source. Defaults to "src_index".
            index_gen_opt (bool, optional): whether use optimized GPU kernel. Defaults to True.

        Inputs:
            score (torch.Tensor): 1-d tensor with label encoding
        """
        super().__init__(**kwargs)
        self.path_num = path_num

    def make_route_decision(self, score):
        hot_mask = torch.nn.functional.one_hot(score, num_classes=self.path_num)
        return hot_mask
