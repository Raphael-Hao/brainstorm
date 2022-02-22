#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /expert_netlet.py
# \brief:
# Author: raphael hao

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

from .base import Netlet
from transformers.activations import ACT2FN

import tvm
import tvm.relay as relay
from tvm import auto_scheduler
from tvm.contrib import graph_runtime

def reset_bias(weight, bias):
    fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    init.uniform_(bias, -bound, bound)


class ExpertNetlet(Netlet):
    expert_num: int
    hidden_size: int
    intermediate_size: int
    dense2_weight: list[torch.Tensor]
    dense2_bias: list[torch.Tensor]

    def __init__(self, config):
        super().__init__(config)
        self.expert_num = config.expert_num
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense1 = nn.linear(self.hidden_size, self.intermediate_size)
        self.dense2 = nn.Linear(self.intermediate_size, self.expert_num)

    def tvm_export(self):
