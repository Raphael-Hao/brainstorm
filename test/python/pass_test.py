# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
from torch import nn
from torch.fx import GraphModule, Graph, Node

import brt
from brt.trace.graph import symbolic_trace
from brt.passes import VerticalFusePass
from brt.router import ScatterRouter, GatherRouter

class PassTest(unittest.TestCase):
    def test_vertical_fuse_pass(self):
        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = ScatterRouter()
                self.expert0 = nn.Sequential(
                    nn.Conv2d(3, 3, 1),
                    nn.ReLU()
                )
                self.expert1 = nn.Sequential(
                    nn.Conv2d(3, 3, 1),
                    nn.ReLU()
                )
                self.gather = GatherRouter()

            def forward(self, x):
                y0, y1 = self.router(x)
                y0 += self.expert0(y0)
                y1 = self.expert1(y1)
                y = self.gather([y0, y1])
                return y

        m = TestModule().eval()
        print(m)
        gm = symbolic_trace(m)
        print(gm)
        g = gm.graph
        print(g)
        vertical_fuse_pass = VerticalFusePass(gm)
        vertical_fuse_pass.run_on_graph()
        gm = vertical_fuse_pass.finalize()
        print(gm)