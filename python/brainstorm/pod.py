#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /pod.py
# \brief: 
# Author: raphael hao
from typing import Any
import torch.nn as nn

class BranchPod(nn.Module):
    def __init__(self, branches: list) -> None:
        super().__init__()
        self.branches = branches

    def forward(self, x):
        return self.branches[x[0]](x[1])