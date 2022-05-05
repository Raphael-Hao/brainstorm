# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Set, Tuple

import brt.nn as nn
import torch
from brt.jit import BlockFuser, CUDACompiler, Templator
from brt.primitive import netlet
from transformers.activations import ACT2FN


@netlet
class FusionExpert(nn.Module):
    kernel_pool = dict()

    def __init__(self, config) -> None:
        super().__init__()
        self.expert_num = config.expert_num
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dense1_weight = []
        dense1_bias = []
        dense2_weight = []
        dense2_bias = []

        for i in range(self.expert_num):
            dense1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
            dense2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)
            dense1_weight.append(dense1.weight.t())
            dense1_bias.append(dense1.bias)
            dense2_weight.append(dense2.weight.t())
            dense2_bias.append(dense2.bias)

        self.dense1_weight = nn.ParameterList(
            [nn.Parameter(dense1_weight[i]) for i in range(self.expert_num)]
        )
        self.dense1_bias = nn.ParameterList(
            [nn.Parameter(dense1_bias[i]) for i in range(self.expert_num)]
        )
        self.dense2_weight = nn.ParameterList(
            [nn.Parameter(dense2_weight[i]) for i in range(self.expert_num)]
        )
        self.dense2_bias = nn.ParameterList(
            [nn.Parameter(dense2_bias[i]) for i in range(self.expert_num)]
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def update(self, tokens_per_expert: Tuple[int, ...]):
        kernel_key = hash(tokens_per_expert)
        if FusionExpert.kernel_pool[kernel_key] is None:
            fusion_templates = []
            for i in range(self.expert_num):
                if tokens_per_expert[i] > 0:
                    fusion_templates.append(
                        Templator.get_template("expert" + tokens_per_expert[i] + ".cu")
                    )
            block_fuser = BlockFuser(fusion_templates)
            expert_code = block_fuser.get_code()
            kernel_func = CUDACompiler.generate_kernel(None, expert_code)
            FusionExpert.kernel_pool[kernel_key] = kernel_func
            self.expert_kernel = kernel_func
        else:
            self.expert_kernel = FusionExpert.kernel_pool[kernel_key]

    def pack_input(self, inputs: torch.Tensor, index: int):
        dense_inputs = []
        if index == 0:
            dense_ouput = torch.empty(
                (inputs.size(0), self.intermediate_size),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            dense_inputs.append(inputs)
            for i in range(self.expert_num):
                dense_inputs.append(self.dense1_weight[i])
            dense_inputs.append(dense_ouput)
            return dense_inputs
        else:
            dense_ouput = torch.empty(
                (inputs.size(0), self.hidden_size),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            dense_inputs.append(inputs)
            for i in range(self.expert_num):
                dense_inputs.append(self.dense2_weight[i])
            dense_inputs.append(dense_ouput)
        return dense_inputs

    def forward(self, hidden_states, token_per_expert: Tuple[int, ...]):
        dense_inputs = self.pack_input(hidden_states, 0)
        self.dense1_kernel(*dense_inputs, extra=[*token_per_expert])
        inter_states = dense_inputs[-1]
        inter_states = self.intermediate_act_fn(inter_states)
        dense_inputs = self.pack_input(inter_states, 1)
        self.dense2_kernel(*dense_inputs, extra=[*token_per_expert])
        inter_states = dense_inputs[-1]
        inter_states = self.dropout(inter_states)
        inter_states = self.LayerNorm(inter_states + hidden_states)
        return inter_states
