# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from config import SwitchTransformersConfig
from transformers.activations import ACT2FN
from brt.jit import make_jit_kernel


class FusedSwitchExpert(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.capacities = config.capacities
        self.ranks = config.ranks
        wi_denses = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_ff, bias=False)
                for _ in range(self.num_experts)
            ]
        )
        wo_denses = nn.ModuleList(
            [
                nn.Linear(config.d_ff, config.d_model, bias=False)
                for _ in range(self.num_experts)
            ]
        )
        sample_inputs = [torch.randn(i, config.d_model) for i in self.capacities]
        self.fused_wi = make_jit_kernel(
            nn.ModuleList(wi_denses),
            sample_inputs,
            opt_level="homo_fuse",
            rank=config.ranks[0],
        )
        sample_inputs = [torch.randn(i, config.d_ff) for i in self.capacities]
        self.fused_wo = make_jit_kernel(
            nn.ModuleList(wo_denses),
            sample_inputs,
            opt_level="homo_fuse",
            rank=config.ranks[1],
        )
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def initialize_weights_from_experts(self, experts: nn.ModuleDict):
        self.fused_wi_standalone_inputs = []
        self.fused_wo_standalone_inputs = []
        for expert in experts.values():
            self.fused_wi_standalone_inputs.extend([expert.wi.weight])
            self.fused_wo_standalone_inputs.extend([expert.wo.weight])
        self.fused_wi_standalone_inputs = nn.ParameterList(
            self.fused_wi_standalone_inputs
        )
        self.fused_wi_standalone_inputs = nn.ParameterList(
            self.fused_wo_standalone_inputs
        )

    def forward(self, dispatched_states):
        capacities = dispatched_states.loads
        print(capacities)
        route_indices = dispatched_states.route_indices
        score = dispatched_states.score
        wi_out = torch.empty(
            (dispatched_states.shape[0], self.d_ff), device=dispatched_states.device
        )
        self.fused_wi(
            shared_inputs=[dispatched_states, wi_out],
            standalone_inputs=self.fused_wi_standalone_inputs,
            capacities=capacities,
        )
        act_out = self.act(wi_out)
        dropout_out = self.dropout(act_out)
        wo_out = torch.empty(
            (dispatched_states.shape[0], self.d_model), device=dispatched_states.device
        )
        self.fused_wo(
            shared_inputs=[dropout_out, wo_out],
            standalone_inputs=self.fused_wo_standalone_inputs,
            capacities=capacities,
        )
        wo_out.route_indices = route_indices
        wo_out.score = score
        wo_out.loads = dispatched_states.loads
        return wo_out


class BatchmamutlSwitchExpert(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def initialize_weights_from_experts(self, experts: nn.ModuleDict):
        fused_wi_weight = []
        fused_wo_weight = []
        for expert in experts.values():
            fused_wi_weight.extend([expert.wi.weight])
            fused_wo_weight.extend([expert.wo.weight])
        fused_wi_weight = (
            torch.cat(fused_wi_weight, dim=0)
            .view(self.num_experts, self.d_ff, self.d_model)
            .permute(0, 2, 1)
        )
        fused_wo_weight = (
            torch.cat(fused_wo_weight, dim=0)
            .view(self.num_experts, self.d_model, self.d_ff)
            .permute(0, 2, 1)
        )
        self.register_parameter("fused_wi_weight", nn.Parameter(fused_wi_weight))
        self.register_parameter("fused_wo_weight", nn.Parameter(fused_wo_weight))

    def forward(self, dispatched_states):

        loads = dispatched_states.loads
        print(loads)
        route_indices = dispatched_states.route_indices
        score = dispatched_states.score
        dispatched_states = dispatched_states.view(self.num_experts, -1, self.d_model)

        wi_out = torch.bmm(dispatched_states, self.fused_wi_weight)
        act_out = self.act(wi_out)
        dropout_out = self.dropout(act_out)
        wo_out = torch.bmm(dropout_out, self.fused_wo_weight)
        wo_out = wo_out.view(-1, self.d_model)
        wo_out.route_indices = route_indices
        wo_out.score = score
        wo_out.loads = loads
        return wo_out
