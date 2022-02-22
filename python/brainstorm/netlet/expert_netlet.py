#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /expert_netlet.py
# \brief:
# Author: raphael hao

from __future__ import annotations
import math
import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from .base import Netlet
from transformers.activations import ACT2FN


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
        self.dense1_weight = [
            torch.empty((self.intermediate_size, self.hidden_size))
            for _ in range(self.expert_num)
        ]
        self.dense1_bias = [
            Parameter(torch.empty(self.intermediate_size))
            for _ in range(self.expert_num)
        ]
        self.dense2_weight = [
            torch.empty((self.hidden_size, self.intermediate_size))
            for _ in range(self.expert_num)
        ]
        self.dense2_bias = [
            Parameter(torch.empty(self.hidden_size)) for _ in range(self.expert_num)
        ]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.expert_num):
            init.kaiming_uniform_(self.dense1_weight[i], a=math.sqrt(5))
            reset_bias(self.dense1_weight[i], self.dense1_bias[i])
            init.kaiming_uniform_(self.dense2_weight[i], a=math.sqrt(5))
            reset_bias(self.dense2_weight[i], self.dense2_bias[i])

    def tvm_export(self):
        class Expert
    
    
    
