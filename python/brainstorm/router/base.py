#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# \file: /router.py
# \brief: base router for brainstorm
# Author: v-weihaocui
# Email: v-weihaocui@microsoft.com
# from typing import List, Dict, OrderedDict
import torch.nn as nn
import torch


class Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @torch.jit.ignore
    def forward(self, input):
        return input, None