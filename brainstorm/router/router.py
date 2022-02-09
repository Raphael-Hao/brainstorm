#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# \file: /router.py
# \brief: base router for brainstorm
# Author: v-weihaocui
# Email: v-weihaocui@microsoft.com

import torch.nn as nn

class Router(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def __repr__(self):
        return self.__class__.__name__ + "()"
