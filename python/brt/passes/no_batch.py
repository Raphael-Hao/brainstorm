# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from brt.passes.base import PassBase, register_pass

@register_pass("no_batch")
class NoBatchPass(PassBase):
    def __init__(self, m: Union[nn.Module, GraphModule]):
        super().__init__(m)

    def run_on_graph(self) -> None:
        pass