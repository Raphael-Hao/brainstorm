# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist

from brt.runtime import log
from brt.router.protocol.base import ProtocolBase, register_protocol
import brt._C.router as c_router

logger = log.get_logger(__file__)


def _get_world_size(group=None):
    try:
        return dist.get_world_size(group)
    except:
        return 1


def _simple_all_reduce(in_data, group=None, op=torch.distributed.ReduceOp.SUM):
    world_size = _get_world_size(group)
    if world_size == 1:
        return in_data
    output = torch.clone(in_data, memory_format=torch.contiguous_format)
    dist.all_reduce(output, op=op, group=group)
    return output


def get_compute_location_func(score_sorted=False, score=None):
    if score_sorted:
        importance_score = -1 * score.max(dim=1)[0]

        def compute_location(one_hot_mask):
            sorted_mask = one_hot_mask[importance_score.argsort(dim=0)]
            dst_indices, loads = c_router.generate_indices_and_loads(sorted_mask)
            return dst_indices[importance_score.argsort(dim=0).argsort(dim=0)], loads

    else:

        def compute_location(one_hot_mask):
            dst_indices, loads = c_router.generate_indices_and_loads(one_hot_mask)
            return dst_indices, loads

    return compute_location


def process_mask(hot_mask, prefix_base, capacity, score_sorted=False, score=None):
    if score_sorted:
        importance_score = -1 * score.max(dim=1)[0]
        importance_indices = importance_score.argsort(dim=0)
        sorted_mask = hot_mask[importance_indices]
        throttled_mask, prefix_base = c_router.throttle_hotmask(
            sorted_mask, prefix_base, capacity
        )
        return throttled_mask[importance_indices.argsort(dim=0)], prefix_base
    else:
        throttled_mask, prefix_base = c_router.throttle_hotmask(
            hot_mask, prefix_base, capacity
        )
        return throttled_mask, prefix_base


def _one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros((data.size(0), num_classes), device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result


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
        **kwargs,
    ):
        super().__init__(**kwargs)
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
            _one_hot_with_dtype(x, num_classes=num_global_experts, dtype=torch.int32)
            for x in indices_s
        ]
        # gates_s = [(score * x).sum(dim=1) for x in masks_se]

        loss = self.generate_auxiliary(score, logits_wo_noise, logits, topk_indices)

        samples_per_expert = (
            int(score.size(0)) + num_global_experts - 1
        ) // num_global_experts

        if self.capacity_factor > 0:
            runtime_capacities = self.top_k * int(
                self.capacity_factor * samples_per_expert
            )
        else:
            raise NotImplementedError("capacity_factor must be greater than 0")
        path_num = masks_se[0].size(1)
        runtime_capacities = torch.tensor(
            [runtime_capacities] * path_num, dtype=torch.int32, device=masks_se[0].device
        )

        hot_mask, _remain_capacities = process_mask(
            masks_se[0], runtime_capacities, self.batch_prioritized_routing, score
        )

        new_score = score

        return hot_mask, runtime_capacities, new_score, loss

    def make_route_decision_legacy(
        self, score: torch.Tensor, logits_wo_noise: torch.Tensor, logits: torch.Tensor
    ):
        num_global_experts = score.size(1)

        topk_indices = torch.topk(score, self.top_k, dim=1).indices

        indices_s = [x.view(-1) for x in topk_indices.chunk(self.top_k, dim=1)]

        masks_se = [
            _one_hot_with_dtype(x, num_classes=num_global_experts, dtype=x.dtype)
            for x in indices_s
        ]
        gates_s = [(score * x).sum(dim=1) for x in masks_se]

        loss = self.generate_auxiliary(score, logits_wo_noise, logits, topk_indices)

        samples_per_expert = (
            int(score.size(0)) + num_global_experts - 1
        ) // num_global_experts

        if self.capacity_factor > 0:
            capacity = self.top_k * int(self.capacity_factor * samples_per_expert)

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

        if self.capacity_factor > 0:
            capacity = self.top_k * int(self.capacity_factor * samples_per_expert)
        else:
            capacity = torch.max(route_indices)
            capacity = int(
                _simple_all_reduce(
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
        loads = acc_base.view(-1)
        loads.capacity = capacity
        capacity = torch.zeros_like(
            loads, dtype=loads.dtype, device=loads.device
        ).fill_(capacity)

        return route_indices, loads, capacity, new_score, loss

    def generate_auxiliary(self, score, logits_wo_noise, logits, topk_ids):
        # if self.is_gshard_loss:
        #     loss = losses.gshard_loss(score, topk_ids)
        # else:
        #     num_global_experts = score.size(1)
        #     loss = losses.load_importance_loss(
        #         F.softmax(logits, dim=1),
        #         logits_wo_noise.gather(index=topk_ids, dim=1),
        #         num_global_experts,
        #         self.gate_noise,
        #     )
        loss = None
        return loss
