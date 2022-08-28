# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import torch
import torch.nn as nn
from brt.runtime.weight_load import WeightLoader

class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.linear(x)
        x = self.conv(x)
        return x

class LoadTest(unittest.TestCase):
    def test_weight_loader(self):
        simple_net =SimpleNet()
        in_data = torch.randn(1,3,10,10)
        origin_out_data = simple_net(in_data)
        WeightLoader.init()
        pinned_simple_net = WeightLoader.pin_memory(simple_net)
        pinned_out_data = pinned_simple_net(in_data)
        cuda_simple_net = WeightLoader.load(pinned_simple_net)
        cuda_out_data = cuda_simple_net(in_data.cuda())
        unload_simple_net = WeightLoader.unload(cuda_simple_net)
        unload_out_data = unload_simple_net(in_data)
        self.assertTrue(torch.allclose(origin_out_data, pinned_out_data))
        self.assertTrue(torch.allclose(origin_out_data, cuda_out_data.cpu()))
        self.assertTrue(torch.allclose(origin_out_data, unload_out_data))