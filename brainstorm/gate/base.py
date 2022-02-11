#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /gating.py
# \brief:
# Author: raphael hao

import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"
