# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, List, Union

import torch
from brt.runtime import log
from brt.router.fabric.base import FabricBase, register_fabric

logger = log.get_logger(__file__)


@register_fabric("zero_skip")
class ZeroSkipFabric(FabricBase):
    def __init__(self, flow_num: int) -> None:
        super().__init__(index_format=None, flow_num=flow_num)

    def forward(self, in_flows, score: torch.Tensor = None):
        path_num = 1
        if score is not None:
            path_num = score.size(1)
        if self.flow_num == 1:
            empty_flows, ret_flows = self._check_empty(in_flows)
            if not empty_flows:
                return False, None
            if path_num == 1:
                return True, ret_flows
            else:
                return True, [ret_flows for _ in range(path_num)]
        else:
            """NOTE: we assume the in_flows will all be the same zero or no-zero
            tensors, therefore we will only check the first in_flow
            """
            empty_flows, ret_flows = self._check_empty(in_flows[0])
            if not empty_flows:
                return False, None
            if path_num == 1:
                return True, [ret_flows for _ in range(self.flow_num)]
            else:
                return True, [
                    [ret_flows for _ in range(path_num)] for _ in range(self.flow_num)
                ]

    def _check_empty(self, in_flows) -> Tuple[bool, Union[None, torch.Tensor]]:
        if isinstance(in_flows, torch.Tensor):
            if in_flows.numel() == 0:
                return True, in_flows
            else:
                return False, None
        if isinstance(in_flows, (Tuple, List)):
            empty_flows = True
            for flow in in_flows:
                empty_flow, ret_flows = self._check_empty(flow)
                empty_flows = empty_flows and empty_flow
                if not empty_flows:
                    return False, None
            return empty_flows, ret_flows
