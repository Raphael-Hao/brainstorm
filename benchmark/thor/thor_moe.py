#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /thor_moe.py
# \brief:
# Author: raphael hao

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class ThorInterOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        inter_states = self.dense1(hidden_states)
        inter_states = self.intermediate_act_fn(inter_states)
        inter_states = self.dense2(inter_states)
        inter_states = self.dropout(inter_states)
        inter_states = self.LayerNorm(inter_states + hidden_states)
        return inter_states


class MaskSerialThorMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expert_num = config.expert_num
        self.dense1 = nn.ModuleList(
            nn.Linear(config.hidden_size, config.intermediate_size)
            for _ in range(self.expert_num)
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.ModuleList(
            nn.Linear(config.intermediate_size, config.hidden_size)
            for _ in range(self.expert_num)
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.token_num = config.token_num
        self.mask = torch.randint(
            0,
            self.expert_num,
            (1, config.token_num, 1),
        )
        self.mask = self.mask.repeat(1, 1, config.hidden_size)
        self.expert_mask = [self.mask.eq(i) for i in range(self.expert_num)]

    def forward(self, hidden_states):
        inter_states = []
        for expert in range(self.expert_num):
            temp = self.dense1[expert](hidden_states)
            temp = self.intermediate_act_fn(temp)
            temp = self.dense2[expert](temp)
            inter_states.append(temp)
        for i in range(self.expert_num):
            inter_states[i] *= self.expert_mask[i]

        inter_states = torch.stack(inter_states, dim=0)
        inter_states = inter_states.sum(dim=0)
        inter_states = self.dropout(inter_states)
        inter_states = self.LayerNorm(inter_states + hidden_states)
        return inter_states


class MaskFusionThorMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.expert_num = config.expert_num
        dense1_weight = torch.empty(
            1, self.expert_num, config.hidden_size, config.intermediate_size
        )
        dense1_bias = torch.empty(1, self.expert_num, 1, config.intermediate_size)
        dense2_weight = torch.empty(
            1, self.expert_num, config.intermediate_size, config.hidden_size
        )
        dense2_bias = torch.empty(1, self.expert_num, 1, config.hidden_size)

        for i in range(self.expert_num):
            dense1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
            dense2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)
            dense1_weight[0, i, :, :], dense1_bias[0, i, :, :] = (
                dense1.weight.t(),
                dense1.bias,
            )
            dense2_weight[0, i, :, :], dense2_bias[0, i, :, :] = (
                dense2.weight.t(),
                dense2.bias,
            )

        dense1_weight = dense1_weight.view(
            self.expert_num, config.hidden_size, config.intermediate_size
        )
        dense1_bias = dense1_bias.view(self.expert_num, 1, config.intermediate_size)
        dense2_weight = dense2_weight.view(
            self.expert_num, config.intermediate_size, config.hidden_size
        )
        dense2_bias = dense2_bias.view(self.expert_num, 1, config.hidden_size)
        self.register_parameter(name="dense1_weight", param=nn.Parameter(dense1_weight))
        self.register_parameter(name="dense1_bias", param=nn.Parameter(dense1_bias))
        self.register_parameter(name="dense2_weight", param=nn.Parameter(dense2_weight))
        self.register_parameter(name="dense2_bias", param=nn.Parameter(dense2_bias))

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mask = torch.randint(0, self.expert_num, (config.token_num,1))
        self.mask = self.mask.repeat(1, config.hidden_size)
        self.expert_mask = [self.mask.eq(i) for i in range(self.expert_num)]

    def forward(self, hidden_states):
        # inter_states = hidden_states.unsqueeze(1)
        inter_states = hidden_states
        inter_states = torch.matmul(inter_states, self.dense1_weight) + self.dense1_bias
        inter_states = self.intermediate_act_fn(inter_states)
        inter_states = torch.matmul(inter_states, self.dense2_weight) + self.dense2_bias
        for i in range(self.expert_num):
            inter_states[i] *= self.expert_mask[i]
        inter_states = inter_states.sum(dim=0, keepdim=True)

        # print(inter_states.size())
        # inter_states = inter_states.sum(dim=1)
        inter_states = self.dropout(inter_states)
        inter_states = self.LayerNorm(inter_states + hidden_states)
        return inter_states


class SparseSerialThorMoE(nn.Module):
    """Sparse serial thor MoE

    Args:
        # TODO currently only support batch size = 1
    """

    def __init__(self, config):
        super().__init__()
        self.expert_num = config.expert_num
        self.dense1 = nn.ModuleList(
            nn.Linear(config.hidden_size, config.intermediate_size)
            for _ in range(self.expert_num)
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.ModuleList(
            nn.Linear(config.intermediate_size, config.hidden_size)
            for _ in range(self.expert_num)
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # TODO currently only support fixed sequence length / tokens
        self.token_num = config.token_num
        self.random_mask = torch.randperm(self.token_num)
        self.restore_mask = torch.arange(0, self.token_num)
        for i in range(self.token_num):
            original_index = self.random_mask[i]
            self.restore_mask[original_index] = i
        assert self.token_num % (self.expert_num) == 0
        self.runtime = config.runtime

    def generate_mask(self, data):
        self.random_mask = torch.randperm(data.size(0), device=data.device)
        self.restore_mask = torch.empty_like(self.random_mask)
        self.restore_mask[self.random_mask] = torch.arange(
            self.random_mask.size(0), device=self.random_mask.device
        )

    def forward(self, hidden_states):
        # hidden_states: B x T x C
        inter_states = hidden_states.view(self.token_num, hidden_states.size(-1))
        # print(inter_states.size())
        if self.runtime:
            self.generate_mask(inter_states)
        # print(inter_states.size())
        inter_states = inter_states[self.random_mask].contiguous()
        inter_states = inter_states.view(
            self.expert_num,
            int(self.token_num / self.expert_num),
            hidden_states.size(-1),
        )
        tmp_inter_states = []
        for i in range(self.expert_num):
            tmp_states = self.dense1[i](inter_states[i])
            tmp_states = self.intermediate_act_fn(tmp_states)
            tmp_states = self.dense2[i](tmp_states)
            tmp_inter_states.append(tmp_states)
        inter_states = torch.stack(tmp_inter_states, dim=0)
        inter_states = inter_states.view(self.token_num, hidden_states.size(-1))
        inter_states = inter_states[self.restore_mask].contiguous()
        inter_states = self.dropout(inter_states)
        inter_states = self.LayerNorm(inter_states + hidden_states)
        return inter_states


class SparseFusionThorMoE(nn.Module):
    """Sparse fusion thor MoE

    Args:
        # TODO currently only support batch size = 1
    """

    def __init__(self, config):
        super().__init__()
        self.expert_num = config.expert_num
        dense1_weight = torch.empty(
            1, self.expert_num, config.hidden_size, config.intermediate_size
        )
        dense1_bias = torch.empty(1, self.expert_num, 1, config.intermediate_size)
        dense2_weight = torch.empty(
            1, self.expert_num, config.intermediate_size, config.hidden_size
        )
        dense2_bias = torch.empty(1, self.expert_num, 1, config.hidden_size)

        for i in range(self.expert_num):
            dense1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
            dense2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)
            dense1_weight[0, i, :, :], dense1_bias[0, i, :, :] = (
                dense1.weight.t(),
                dense1.bias,
            )
            dense2_weight[0, i, :, :], dense2_bias[0, i, :, :] = (
                dense2.weight.t(),
                dense2.bias,
            )

        dense1_weight = dense1_weight.view(
            self.expert_num, config.hidden_size, config.intermediate_size
        )
        dense1_bias = dense1_bias.view(self.expert_num, 1, config.intermediate_size)
        dense2_weight = dense2_weight.view(
            self.expert_num, config.intermediate_size, config.hidden_size
        )
        dense2_bias = dense2_bias.view(self.expert_num, 1, config.hidden_size)
        self.register_parameter(name="dense1_weight", param=nn.Parameter(dense1_weight))
        self.register_parameter(name="dense1_bias", param=nn.Parameter(dense1_bias))
        self.register_parameter(name="dense2_weight", param=nn.Parameter(dense2_weight))
        self.register_parameter(name="dense2_bias", param=nn.Parameter(dense2_bias))

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # TODO currently only support fixed sequence length / tokens
        self.token_num = config.token_num
        self.random_mask = torch.randperm(self.token_num)
        self.restore_mask = torch.arange(0, self.token_num)
        for i in range(self.token_num):
            original_index = self.random_mask[i]
            self.restore_mask[original_index] = i
        assert self.token_num % (self.expert_num) == 0
        self.runtime = config.runtime

    def generate_mask(self, data):
        self.random_mask = torch.randperm(data.size(0), device=data.device)
        self.restore_mask = torch.empty_like(self.random_mask)
        self.restore_mask[self.random_mask] = torch.arange(
            self.random_mask.size(0), device=self.random_mask.device
        )

    def forward(self, hidden_states):
        # hidden_states: B x T x C
        inter_states = hidden_states.view(-1, hidden_states.size(-1))
        if self.runtime:
            self.generate_mask(inter_states)
        inter_states = inter_states[self.random_mask].contiguous()
        inter_states = inter_states.view(self.expert_num, -1, hidden_states.size(-1))
        inter_states = torch.matmul(inter_states, self.dense1_weight) + self.dense1_bias
        inter_states = self.intermediate_act_fn(inter_states)
        inter_states = torch.matmul(inter_states, self.dense2_weight) + self.dense2_bias
        inter_states = inter_states.view(-1, hidden_states.size(-1))
        inter_states = inter_states[self.restore_mask].contiguous()
        inter_states = self.dropout(inter_states)
        inter_states = self.LayerNorm(inter_states + hidden_states)
        return inter_states
