# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.distributed as dist
from brt.router import ScatterRouter, GatherRouter
from configuration_bert_generation import BertGenerationConfig
from transformers.activations import ACT2FN


class BertGenerationMoE(nn.Module):
    def __init__(self, config: BertGenerationConfig, task_locality=False):
        super().__init__()
        self.task_locality = task_locality
        self.placement_aware = config.placement_aware
        if config.placement_aware:
            if task_locality:
                self.task_sactter = ScatterRouter(
                    protocol_type="task",
                    protocol_kwargs={
                        "num_tasks": config.num_tasks,
                    },
                    fabric_type="distributed_placement_dispatch",
                    fabric_kwargs={"task_locality": True},
                )
            self.hash_scatter = ScatterRouter(
                protocol_type="hash",
                protocol_kwargs={
                    "num_tasks": config.num_tasks,
                    "placement_aware": config.placement_aware,
                },
                fabric_type="distributed_placement_combine",
            )
            self.hash_gather = GatherRouter(
                fabric_type="distributed_fused_combine",
                fabric_kwargs={"transform": False},
            )
        else:
            self.hash_scatter = ScatterRouter(
                protocol_type="hash",
                protocol_kwargs={
                    "num_tasks": config.num_tasks,
                },
                fabric_type="distributed_fused_dispatch",
            )
            self.hash_gather = GatherRouter(
                fabric_type="distributed_fused_combine",
                fabric_kwargs={"transform": False},
            )
        self.local_experts = config.num_tasks // dist.get_world_size()
        self.intermediate_denses = nn.ModuleList(
            nn.Linear(config.hidden_size, config.intermediate_size)
            for _ in range(self.local_experts)
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fns = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fns = config.hidden_act
        self.output_denses = nn.ModuleList(
            nn.Linear(config.intermediate_size, config.hidden_size)
            for _ in range(self.local_experts)
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            for _ in range(self.local_experts)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(config.hidden_dropout_prob) for _ in range(self.local_experts)
        )

    def placement_forward(self, x, task_ids):
        if self.task_locality:
            x = self.task_sactter(x, task_ids)
            task_ids = x.score
        x = self.hash_scatter(x, task_ids)
        in_loads = x.in_loads
        out_loads = x.out_loads
        route_indices = x.route_indices
        outputs = []
        base_load_idx = 0
        base_x_idx = 0
        world_size = dist.get_world_size()
        for i in range(self.local_experts):
            load = out_loads[base_load_idx : base_load_idx + world_size].sum().item()
            outputs.append(self.expert_forward(x[base_x_idx : base_x_idx + load], i))
        x = torch.cat(outputs, dim=0)
        x.in_loads = in_loads
        x.out_loads = out_loads
        x.route_indices = route_indices
        x.score = task_ids
        x = self.hash_gather(x)
        return x

    def common_forward(self, x, task_ids):
        x = self.hash_scatter(x, task_ids)
        in_loads = x.in_loads
        out_loads = x.out_loads
        route_indices = x.route_indices
        world_size = dist.get_world_size()
        x_origin_shape = x.shape
        x = x.reshape(world_size, self.local_experts, -1, x.shape[-1])
        xs = (
            x.permute(1, 0, 2, 3)
            .contiguous()
            .view(-1, x.shape[-2], x.shape[-1])
            .chunk(self.local_experts)
        )
        outputs = []
        for i in range(self.local_experts):
            outputs.append(self.expert_forward(x[i], i))
        x = torch.cat(outputs, dim=0)
        x.in_loads = in_loads
        x.out_loads = out_loads
        x.route_indices = route_indices
        x.score = task_ids
        x = self.hash_gather(x)
        return x

    def expert_forward(self, x, expert_idx):
        y = self.intermediate_denses[expert_idx](x)
        y = self.intermediate_act_fns(y)
        y = self.output_denses[expert_idx](y)
        y = self.dropouts[expert_idx](y)
        y = self.layer_norms[expert_idx](y + x)
        return y

    def forward(self, x, task_ids):
        if self.placement_aware:
            x = self.placement_forward(x, task_ids)
        else:
            x = self.common_forward(x, task_ids)
        return x
