# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from brt.passes.base import PassBase, register_pass


@register_pass("analyze")
class AnalyzePass(PassBase):
    def __init__(self, m: Union[nn.Module, GraphModule], origin_m, evaluator):
        super().__init__(m)
        self.origin_m = origin_m
        self.evaluator = evaluator

    def run_on_graph(self) -> None:
        pass
