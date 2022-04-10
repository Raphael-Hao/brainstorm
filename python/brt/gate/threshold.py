#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /threshold.py
# \brief:
# Author: raphael hao
from .base import Gate
import torch


class ThresholdGate(Gate):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, *input):
        return torch.where(
            input[0] > self.threshold,
            torch.ones_like(input[0]),
            torch.zeros_like(input[0]),
        )
