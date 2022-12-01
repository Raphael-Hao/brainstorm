# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from brt.router import ScatterRouter, GatherRouter
from brt.router.protocol.hashed import HashProtocol
from configuration_bert_generation import BertGenerationConfig
from transformers.activations import ACT2FN

USE_EINSUM = False


def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == "s,se->se":
        return a.reshape(a.shape[0], -1) * b
    elif rule == "se,sc->sec":
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == "se,se->s":
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == "sec,sm->ecm":
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == "sec,ecm->sm":
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == "ks,ksm->sm":
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


class BertGenerationMoE(nn.Module):
    def __init__(self, config: BertGenerationConfig, task_locality=False, seed=0):
        super().__init__()
        self.task_locality = task_locality
        self.placement_aware = config.placement_aware
        self.pt_native = config.pt_native
        self.num_tasks = config.num_tasks
        if self.pt_native:
            self.protocol = HashProtocol(num_tasks=config.num_tasks, seed=seed)
        else:
            if config.placement_aware:
                if task_locality:
                    self.task_sactter = ScatterRouter(
                        protocol_type="task",
                        protocol_kwargs={
                            "num_tasks": config.num_tasks,
                            "index_format": "dst_index",
                        },
                        fabric_type="distributed_placement_dispatch",
                        fabric_kwargs={"task_locality": True},
                    )
                self.hash_scatter = ScatterRouter(
                    protocol_type="hash",
                    protocol_kwargs={
                        "num_tasks": config.num_tasks,
                        "placement_aware": config.placement_aware,
                        "index_format": "dst_index",
                    },
                    fabric_type="distributed_placement_dispatch",
                )
                self.hash_gather = GatherRouter(
                    fabric_type="distributed_placement_combine",
                    fabric_kwargs={"transform": False},
                )
            else:
                self.hash_scatter = ScatterRouter(
                    protocol_type="hash",
                    protocol_kwargs={
                        "num_tasks": config.num_tasks,
                        "index_format": "dst_index",
                        "seed": seed,
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

    def placement_forward(self, x: torch.Tensor, task_ids: torch.Tensor):
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
            base_load_idx += world_size
            base_x_idx += load
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
        x = x.reshape(world_size, -1, x.shape[-2], x.shape[-1])
        x = x.permute(1, 0, 2, 3).contiguous().view(-1, x.shape[-2], x.shape[-1])
        xs = x.chunk(self.local_experts, dim=0)
        outputs = []
        for i in range(self.local_experts):
            outputs.append(self.expert_forward(xs[i], i))
        x = torch.cat(outputs, dim=0)
        x.in_loads = in_loads
        x.out_loads = out_loads
        x.route_indices = route_indices
        x.score = task_ids
        x = self.hash_gather(x)
        return x

    def pt_forward(self, x: torch.Tensor, task_ids: torch.Tensor):
        hash_dest = self.protocol.hash_indices[task_ids]
        all_hot_mask = None
        hot_masks = []
        locations = []
        for i in range(hash_dest.size(1)):
            dest = hash_dest[:, i].unsqueeze(1)
            hot_masks.append(
                torch.zeros(
                    (task_ids.size(0), self.num_tasks),
                    dtype=torch.int64,
                    device=task_ids.device,
                ).scatter_(1, dest, 1)
            )
            locations.append((torch.cumsum(hot_masks[-1], dim=0) - 1) * hot_masks[-1])
            all_hot_mask = (
                hot_masks[-1] if all_hot_mask is None else all_hot_mask + hot_masks[-1]
            )

        capacity = all_hot_mask.sum(dim=0).max()
        dist.all_reduce(capacity, op=dist.ReduceOp.MAX)
        world_size = dist.get_world_size()
        x_shape = x.shape
        seq_num = x.shape[0]
        dispatch_mask = None

        for i in range(dest.size(1)):
            mask = hot_masks[i]
            new_mask = torch.zeros(
                seq_num, capacity, dtype=torch.int64, device=x.device
            ).scatter_(1, locations[i], 1)
            tmp_mask = einsum("se,sc->sec", mask, new_mask)
            dispatch_mask = (
                tmp_mask if dispatch_mask is None else dispatch_mask + tmp_mask
            )
        x = x.reshape(seq_num, -1)
        x = einsum("sec,sm->ecm", dispatch_mask.type_as(x), x)
        x = x.reshape(world_size, -1)
        out_x = torch.empty_like(x)
        dist.all_to_all_single(out_x, x)
        out_x = out_x.reshape(world_size, -1, x_shape[-2], x_shape[-1])
        out_x = (
            out_x.permute(1, 0, 2, 3).contiguous().view(-1, x_shape[-2], x_shape[-1])
        )
        xs = out_x.chunk(self.local_experts, dim=0)
        outputs = []
        for i in range(self.local_experts):
            outputs.append(self.expert_forward(xs[i], i))
        out_x = torch.cat(outputs, dim=0)
        out_y = torch.empty_like(out_x)
        dist.all_to_all_single(out_y, out_x)
        out_y = out_y.view(capacity * world_size * self.local_experts, -1)
        out_y = einsum("sec,ecm->sm", dispatch_mask.type_as(out_y), out_y)
        x = out_y.reshape(x_shape)

        return x

    def expert_forward(self, x, expert_idx):
        y = self.intermediate_denses[expert_idx](x)
        y = self.intermediate_act_fns(y)
        y = self.output_denses[expert_idx](y)
        y = self.dropouts[expert_idx](y)
        y = self.layer_norms[expert_idx](y + x)
        return y

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor):
        x = x.contiguous()
        if self.pt_native:
            x = self.pt_forward(x, task_ids)
        else:
            if self.placement_aware:
                x = self.placement_forward(x, task_ids)
                task_ids = x.score
            else:
                x = self.common_forward(x, task_ids)
        return x, task_ids
