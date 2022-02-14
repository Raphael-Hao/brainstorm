#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /top_k.py
# \brief: 
# Author: raphael hao

from .base import Gate
import torch

class TopKGate(Gate):
    def __init__(self, k: int = 1):
        super().__init__()
        self.k = k

    def forward(self, *input):
        return torch.topk(input[0], self.k, dim=1)[1]