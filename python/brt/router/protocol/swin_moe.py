# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tutel.impls import losses, communicate

from brt.runtime import log
from brt.router.protocol.base import ProtocolBase, register_protocol
from brt.router.utils import generate_dst_indices

logger = log.get_logger(__file__)


def get_compute_location_func(sorted=False, score=None):
    if sorted:
        importance_score = -1 * score.max(dim=1)[0]

        def compute_location(one_hot_mask):
            sorted_mask = one_hot_mask[importance_score.argsort(dim=0)]
            dst_indices, loads = generate_dst_indices(sorted_mask)
            return dst_indices[importance_score.argsort(dim=0).argsort(dim=0)], loads

    else:

        def compute_location(one_hot_mask):
            dst_indices, loads = generate_dst_indices(one_hot_mask)
            return dst_indices, loads

    return compute_location


@register_protocol("swin_moe")
class SwinMoEProtocol(ProtocolBase):
    def __init__(
        self,
        top_k=2,
        capacity_factor=0,
        gate_noise=0.0,
        group: dist.group = None,
        is_postscore=True,
        batch_prioritized_routing=False,
        normalize_gate=True,
        is_gshard_loss=True,
        alignment=1,
        index_format="dst_index",
        index_gen_opt=True,
    ):
        super().__init__(index_format=index_format, index_gen_opt=index_gen_opt)
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate_noise = gate_noise
        self.normalize_gate = normalize_gate
        self.batch_prioritized_routing = batch_prioritized_routing
        self.is_postscore = is_postscore
        self.group = group
        self.is_gshard_loss = is_gshard_loss
        self.alignment = alignment

    def make_route_decision(
        self, score: torch.Tensor, logits_wo_noise: torch.Tensor, logits: torch.Tensor
    ):
        num_global_experts = score.size(1)

        topk_indices = torch.topk(score, self.top_k, dim=1).indices

        indices_s = [x.view(-1) for x in topk_indices.chunk(self.top_k, dim=1)]

        masks_se = [
            losses._one_hot_with_dtype(x, num_classes=num_global_experts, dtype=x.dtype)
            for x in indices_s
        ]
        gates_s = [(score * x).sum(dim=1) for x in masks_se]

        loss = self.generate_auxiliary(score, logits_wo_noise, logits, topk_indices)

        self.compute_location = get_compute_location_func(
            self.batch_prioritized_routing, score
        )

        locations_1, loads_1 = self.compute_location(masks_se[0])

        route_indices = locations_1.to(torch.int32)

        new_score = score
        if self.top_k > 1:
            acc_base = None
            for k in range(1, self.top_k):
                acc_base = (
                    loads_1.unsqueeze(0)
                    if acc_base is None
                    else acc_base + loads_2.unsqueeze(0)
                )
                locations_2, loads_2 = self.compute_location(masks_se[k])
                locations_2 = ((locations_2 + acc_base) * masks_se[k]).to(torch.int32)
                route_indices += locations_2

            acc_base += loads_2.unsqueeze(0)

            if self.normalize_gate:
                denom_s = torch.clamp(
                    sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps
                )
                gates_s = [x / denom_s for x in gates_s]

                new_score = torch.zeros_like(
                    score, dtype=score.dtype, device=score.device
                )

                for indices, gates in zip(indices_s, gates_s):
                    new_score.scatter_(1, indices.unsqueeze(-1), gates.unsqueeze(-1))
        else:
            acc_base = loads_1

        samples_per_expert = (
            int(score.size(0)) + num_global_experts - 1
        ) // num_global_experts
        if self.capacity_factor > 0:
            capacity = self.top_k * int(self.capacity_factor * samples_per_expert)
        else:
            capacity = torch.max(route_indices)
            capacity = int(
                communicate.simple_all_reduce(
                    capacity, group=self.group, op=torch.distributed.ReduceOp.MAX
                )
            )
            if self.capacity_factor < 0:
                capacity = min(
                    capacity,
                    self.top_k
                    * int(
                        -self.capacity_factor
                        * (
                            (int(score.size(0)) + num_global_experts - 1)
                            // num_global_experts
                        )
                    ),
                )

        remainder = capacity % self.alignment
        if remainder > 0:
            capacity = capacity + self.alignment - remainder

        loads = acc_base.view(-1).cpu()
        capacity = torch.zeros_like(loads, dtype=loads.dtype, device=loads.device).fill_(capacity)

        return route_indices, acc_base.view(-1), capacity, new_score, loss

    def generate_auxiliary(self, score, logits_wo_noise, logits, topk_ids):
        if self.is_gshard_loss:
            loss = losses.gshard_loss(score, topk_ids)
        else:
            num_global_experts = score.size(1)
            loss = losses.load_importance_loss(
                F.softmax(logits, dim=1),
                logits_wo_noise.gather(index=topk_ids, dim=1),
                num_global_experts,
                self.gate_noise,
            )
        return loss
