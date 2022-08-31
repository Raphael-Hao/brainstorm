# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.passes.base import PassBase, register_pass
from torch.fx.graph import Graph
import torch.nn as nn


@register_pass("horiz_fuse")
class HorizFusePass(PassBase):
    jit_need_modules = [nn.Conv2d, nn.Linear]
    native_modules = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax]

    def run_on_graph(self):
        pass