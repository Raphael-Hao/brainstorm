#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /random.py
# \brief:
# Author: raphael hao

from .base import Gate
import torch


class RandomGate(Gate):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *input):
        return torch.randint(0, input[0].shape[1], (input[0].shape[0],))
