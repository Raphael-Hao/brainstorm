#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /pod.py
# \brief: 
# Author: raphael hao

import torch
import torch.nn as nn
from .netlet import Netlet
from .router import router

class Pod(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError