# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import brt
import brt.nn as nn
import torch
from brt.router import RandomGatherRouter, RandomScatterRouter


@brt.domain
class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scatter_router = RandomScatterRouter(route_num=2)
        self.expert1 = nn.Linear(10, 10)
        self.expert2 = nn.Linear(10, 10)
        self.gather_router = RandomGatherRouter(route_num=2)

    def forward(self, x):
        route_results, reverse_indice, reverse_shape = self.scatter_router(x)
        x_0 = self.expert1(route_results[0])
        x_1 = self.expert2(route_results[1])
        x = self.gather_router([x_0, x_1], reverse_indice, reverse_shape)
        return x

class RouterTest(unittest.TestCase):
    def test_forward(self):
        moe_model = MoE()
        x = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        x = moe_model(x)
        return x
    def test_jit(self):
        moe_model = MoE()
        try:
            script_simple_net = torch.jit.script(moe_model)
            script_simple_net.inlined_graph
        except Exception as e:
            self.fail(f"Failed to inline the jitted graph: {e}")
    def test_traced(self):
        pass
