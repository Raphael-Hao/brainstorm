# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import brt
import brt.nn as nn
import torch


class NetletTest(unittest.TestCase):
    def test_none_input(self):
        @brt.netlet
        class SimpleNetlet(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x
        simple_net = SimpleNetlet()
        try:
            simple_net(None)
        except Exception as e:
            self.fail(f"Netlet failed to accept None as input: {e}")
        
    def test_jit(self):
        @brt.netlet
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        @brt.domain
        class SimpleNet2(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = SimpleNet()

            def forward(self, x):
                x = self.net(x)
                return x

        simple_net2 = SimpleNet2()
        try:
            script_simple_net = torch.jit.script(simple_net2)
            script_simple_net.inlined_graph
        except Exception as e:
            self.fail(f"Failed to inline the jitted graph: {e}")

    def test_graph_build(self):
        pass
