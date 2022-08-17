# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.passes.base import PassBase, register_pass
from torch.fx.graph import Graph
import torch.nn as nn


@register_pass("horiz_fuse")
class HorizFusePass(PassBase):
    jit_need_modules = [nn.Conv2d, nn.Linear]
    native_modules = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax]

    @classmethod
    def run_on_graph(cls, graph):
        for module in graph.modules():
            if isinstance(module, cls.jit_need_modules):
                module.fuse_parameters()
            if isinstance(module, cls.native_modules):
                module.fuse_parameters()
