#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# \file: /router.py
# \brief: base router for brainstorm
# Author: v-weihaocui
# Email: v-weihaocui@microsoft.com

import torch

class Router:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def __repr__(self):
        return self.__class__.__name__ + "()"
