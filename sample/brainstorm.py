#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /brainstorm.py
# \brief:
# Author: raphael hao
from typing import List, Optional, Union

import torch.nn as nn
import torch


class shutter(nn.Module):
    def __init__(self) -> None:
        self.token_grain = 16

    def forward(self, x):
        return [x for _ in range(self.token_grain)]


class SampleDNN(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.shutter = shutter()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


x: Union[torch.Tensor, List[torch.Tensor]]
Linear: nn.Module
ScatterRouter: nn.Module
GatherRouter: nn.Module
Shutter: nn.Module
Netlet: nn.Module
RandomRouter: nn.Module
CandidateModules: nn.ModuleList
DistScatterRouter: nn.Module
DistGatherRouter: nn.Module
RecursiveRouter: nn.Module
FusionNetlet: nn.Module
NetletGroup: nn.ModuleDict

x = Linear(x)
x = Shutter(x)
x = RandomRouter(x, CandidateModules)
x = Linear(x)

# transform to our IR
x = Linear(x)
x = Shutter(x)
x, y = ScatterRouter(x) # x the scatter result, y the reverse indices
x = NetletGroup(x)
x = GatherRouter(x, y)
x = Linear(x)

# determine single or distribute acording to hint
x = Linear(x)
x = Shutter(x)
x, y = DistScatterRouter(x)
x = Netlet(x)
x = DistGatherRouter(x, y)

# horizontal fusion
x = Linear(x)
x = Shutter(x)
x, y = DistScatterRouter(x)
x = FusionNetlet(x)
x = DistGatherRouter(x, y)


# recursive router
x = Linear(x)
x = Shutter(x)
with RecursiveRouter(x) as r:
    x = r.run(x)