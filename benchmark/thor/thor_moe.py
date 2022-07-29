#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /thor_moe.py
# \brief:
# Author: raphael hao

import torch
import torch.nn as nn
from brt.runtime import BRT_KERNEL_TEMPLATE_PATH
from brt.jit import make_jit_kernel
from brt.runtime.proto_tensor import collect_proto_attr_stack, init_proto_tensor
from brt.app import (
    RandScatter,
)
from brt.router import GatherRouter

from transformers.activations import ACT2FN


class ThorInterOutput(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

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


class ThorExpert(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, inter_state):
        x = self.dense1(inter_state)
        x = self.intermediate_act_fn(x)
        x = self.dense2(x)
        return x


class FusedThorExpert(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dense1s = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.intermediate_size)
                for _ in range(config.expert_num)
            ]
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        dense2s = nn.ModuleList(
            [
                nn.Linear(config.intermediate_size, config.hidden_size)
                for _ in range(config.expert_num)
            ]
        )
        capacities = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        sample_inputs = [torch.ones(i, config.hidden_size) for i in capacities]

        self.expert1_kernel = make_jit_kernel(
            dense1s, sample_inputs, opt_level="homo_fuse"
        )

        sample_inputs = [torch.ones(i, config.intermediate_size) for i in capacities]

        self.expert2_kernel = make_jit_kernel(
            dense2s, sample_inputs, opt_level="homo_fuse"
        )

        self.expert1_standalone_inputs = []
        for linear in dense1s:
            self.expert1_standalone_inputs.extend([linear.weight, linear.bias])
        self.expert1_standalone_inputs = nn.ParameterList(
            self.expert1_standalone_inputs
        )
        self.expert2_standalone_inputs = []
        for linear in dense2s:
            self.expert2_standalone_inputs.extend([linear.weight, linear.bias])
        self.expert2_standalone_inputs = nn.ParameterList(
            self.expert2_standalone_inputs
        )

    def forward(
        self,
        inter_state: torch.Tensor,
    ) -> torch.Tensor:
        capacities = inter_state.load.tolist()
        tag_stack, load_stack, extra_stack = collect_proto_attr_stack(inter_state)
        expert1_out = torch.zeros(
            inter_state.shape[0], self.intermediate_size, device=inter_state.device
        )
        self.expert1_kernel(
            shared_inputs=[inter_state, expert1_out],
            standalone_inputs=self.expert1_standalone_inputs,
            capacities=capacities,
        )
        x = self.intermediate_act_fn(expert1_out)
        expert2_out = torch.zeros(
            inter_state.shape[0], self.hidden_size, device=inter_state.device
        )
        self.expert2_kernel(
            shared_inputs=[x, expert2_out],
            standalone_inputs=self.expert2_standalone_inputs,
            capacities=capacities,
        )
        expert2_out = init_proto_tensor(expert2_out, tag_stack, load_stack, extra_stack)
        return expert2_out


class ThorMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [ThorExpert(config) for _ in range(config.expert_num)]
        )
        self.scatter_router = RandScatterRouter(dst_num=config.expert_num)
        self.gather_router = RandGatherRouter(dst_num=config.expert_num)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # B x T x H -> T x H
        inter_states = hidden_states.view(-1, hidden_states.size(-1))
        route_results = self.scatter_router(inter_states)
        expert_results = []
        for i, expert in enumerate(self.experts):
            expert_results.append(expert(route_results[i]))
        inter_states = self.gather_router(expert_results)
        # T x H -> B x T x H
        inter_states = inter_states.view(
            hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
        )
        x = self.layer_norm(inter_states + hidden_states)
        return x


class FusedThorMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.scatter_router = RandHomoFusedScatterRouter(
            dst_num=config.expert_num,
            supported_capacities=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        )
        self.gather_router = RandHomoFusedGatherRouter(dst_num=config.expert_num)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fused_expert = FusedThorExpert(config)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.stream = torch.cuda.default_stream()

    def forward(self, hidden_states):
        # B x T x H -> T x H
        inter_states = hidden_states.view(-1, hidden_states.size(-1))
        # self.start_event.record(stream=self.stream)
        x = self.scatter_router(inter_states)
        # self.end_event.record(stream=self.stream)
        # self.stream.synchronize()
        # print("fused scatter time: ", self.start_event.elapsed_time(self.end_event))

        # self.start_event.record(stream=self.stream)
        x = self.fused_expert(x)
        # self.end_event.record(stream=self.stream)
        # self.stream.synchronize()
        # print("fused expert time: ", self.start_event.elapsed_time(self.end_event))

        # self.start_event.record(stream=self.stream)
        inter_states = self.gather_router(x)
        # self.end_event.record(stream=self.stream)
        # self.stream.synchronize()
        # print("gather router time: ", self.start_event.elapsed_time(self.end_event))

        # T x H -> B x T x H
        # self.start_event.record(stream=self.stream)
        inter_states = inter_states.view(
            hidden_states.size(0), hidden_states.size(1), hidden_states.size(2)
        )
        x = self.layer_norm(inter_states + hidden_states)
        # self.end_event.record(stream=self.stream)
        # self.stream.synchronize()
        # print("layer norm time: ", self.start_event.elapsed_time(self.end_event))

        return x


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
        self.mask = torch.randint(0, self.expert_num, (config.token_num, 1))
        self.mask = self.mask.repeat(1, config.hidden_size)
        self.expert_mask = [self.mask.eq(i) for i in range(self.expert_num)]

    def forward(self, hidden_states):
        # inter_states = hidden_states.unsqueeze(1)
        inter_states = hidden_states
        inter_states = torch.matmul(inter_states, self.dense1_weight) + self.dense1_bias
        inter_states = self.intermediate_act_fn(inter_states)
        inter_states = torch.matmul(inter_states, self.dense2_weight) + self.dense2_bias
        tmp_inter_states = []
        for i in range(self.expert_num):
            tmp_states = inter_states[i] * self.expert_mask[i]
            tmp_inter_states.append(tmp_states)
        inter_states = torch.stack(tmp_inter_states, dim=0)
        inter_states = inter_states.sum(dim=0, keepdim=True)

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
        if self.runtime:
            self.generate_mask(inter_states)
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
