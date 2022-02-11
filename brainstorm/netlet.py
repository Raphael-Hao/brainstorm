#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /netlet.py
# \brief: 
# Author: raphael hao
import torch
import torch.nn as nn

class Netlet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError