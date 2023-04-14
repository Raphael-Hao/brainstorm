# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Union

import torch.nn as nn
from brt.passes.base import PassBase, register_pass
from torch.fx import GraphModule


@register_pass("trace")
class TracePass(PassBase):
    def __init__(self, m: Union[nn.Module, GraphModule]):
        super().__init__(m)

    def run_on_graph(self) -> None:
        pass
