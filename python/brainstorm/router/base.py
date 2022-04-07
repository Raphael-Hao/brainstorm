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
        self.active_counter = 0

    @torch.jit.ignore
    def forward(self, input):
        return self.route(input)

    def route(self, input):
        raise NotImplementedError

    def record(self):
        raise NotImplementedError

    def register_router(self):
        pass
