# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import log
from brt.router.utils import generate_indices
from brt.router.protocol.base import ProtocolBase, register_protocol

__all__ = ["LabelProtocol"]

logger = log.get_logger(__file__)


@register_protocol("label")
class LabelProtocol(ProtocolBase):
    def __init__(
        self,
        flow_num: int,
        supported_capacities: torch.Tensor = None,
        index_format="src_index",
        index_gen_opt=True,
    ):
        """Label protocol

        Args:
            supported_capacities (optional): _description_. Defaults to None.
            index_format (str, optional): index tensors according to destination or source. Defaults to "src_index".
            index_gen_opt (bool, optional): whether use optimized GPU kernel. Defaults to True.
        
        Inputs:
            score (torch.Tensor): 1-d tensor with label encoding
        """
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.supported_capacities = supported_capacities
        self.flow_num = flow_num

    def make_route_decision(self, score):
        hot_mask = torch.nn.functional.one_hot(score, num_classes=self.flow_num)
        route_indices, loads = generate_indices(
            hot_mask, self.supported_capacities, self.index_format, self.index_gen_opt
        )
        return route_indices, loads, loads

    def update(self, supported_capacities):
        self.supported_capacities = supported_capacities
