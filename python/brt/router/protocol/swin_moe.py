# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions.normal import Normal
from tutel.jit_kernels.gating import fast_cumsum_sub_one

from brt.common import log
from brt.router.protocol.base import ProtocolBase, register_protocol

logger = log.get_logger(__file__)


def one_hot_with_dtype(data, num_classes, dtype):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), 1)
    return result


class PrimFwdAllgather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input_shape = input.shape
        num_nodes = dist.get_world_size()
        output = torch.empty(
            [num_nodes, input.numel()], device=input.device, dtype=input.dtype
        )
        tensor_list = [
            x.contiguous() for x in torch.chunk(output, chunks=num_nodes, dim=0)
        ]
        dist.all_gather(tensor_list=tensor_list, tensor=input.contiguous())
        output = output.view([input.shape[0] * num_nodes] + list(input.shape[1:]))
        return output

    @staticmethod
    def backward(ctx, doutput):
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
        dinput = (
            doutput.contiguous()
            .view(world_size, -1)
            .contiguous()[world_rank, :]
            .view(ctx.input_shape)
            .contiguous()
        )
        return dinput


def load_balance(gates, mask1, num_global_experts, fp32_gate):
    if gates.dtype == torch.float32 or fp32_gate:
        me = torch.sum(gates.float(), dim=0)
        ce = torch.sum(mask1.to(me.dtype), dim=0)
        l_loss = torch.sum(me * ce) * (
            num_global_experts / (gates.size(0) * gates.size(0))
        )
    else:
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.to(gates.dtype), dim=0)
        l_loss = torch.sum(me * ce) * num_global_experts
    return l_loss


def importance_loss(gates_wo_noise):
    Impi = gates_wo_noise.float().sum(0)
    l_imp = Impi.float().var() / (Impi.float().mean() ** 2 + 1e-10)

    return l_imp


def vitmoe_load_loss(gates_wo_noise, topk_logits, num_global_experts):
    normal = Normal(
        torch.tensor([0.0], device=gates_wo_noise.device),
        torch.tensor([1.0 / num_global_experts], device=gates_wo_noise.device),
    )
    threshold = topk_logits[:, -1].view(-1, 1).float()
    diff = gates_wo_noise.float() - threshold.float()
    prob = normal.cdf(diff)
    Load = prob.sum(0)
    l_load = Load.float().var() / (Load.float().mean() ** 2 + 1e-10)

    return l_load


def vitmoe_loss(gates_wo_noise, topk_logits, num_global_experts, use_global_loss=False):
    if use_global_loss:
        _gates_wo_noise = PrimFwdAllgather.apply(gates_wo_noise)
        _topk_logits = PrimFwdAllgather.apply(topk_logits)
    else:
        _gates_wo_noise = gates_wo_noise
        _topk_logits = topk_logits
    l_imp = importance_loss(_gates_wo_noise)
    l_load = vitmoe_load_loss(_gates_wo_noise, _topk_logits, num_global_experts)
    return (l_imp + l_load) / 2.0


def get_compute_location_func(sorted=False, importance=None):
    if sorted:
        importance = -1 * importance.max(dim=1)[0]

        def compute_location(one_hot_mask):
            sorted_mask = one_hot_mask[importance.argsort(dim=0)]
            sorted_cumsum = fast_cumsum_sub_one(sorted_mask) * sorted_mask
            return sorted_cumsum[importance.argsort(dim=0).argsort(dim=0)]

    else:

        def compute_location(one_hot_mask):
            return fast_cumsum_sub_one(one_hot_mask) * one_hot_mask

    return compute_location


@register_protocol("swin_moe")
class SwinMoEProtocol(ProtocolBase):
    def __init__(self, **kwargs):
        super().__init__()
        top_k = kwargs.get("top_k")
        capacity_factor = kwargs.get("capacity_factor")
        num_global_experts = kwargs.get("num_global_experts")
        self.top_k = min(top_k, num_global_experts)
        self.capacity_factor = float(os.environ.get("CAP_FACTOR", capacity_factor))
        self.num_global_experts = num_global_experts
        self.normalize_gate = kwargs.get("normalize_gate")
        self.vitmoe_loss = kwargs.get("vitmoe_loss")
        self.use_noise = kwargs.get("use_noise")
        if self.vitmoe_loss:
            logger.warning(
                "change use_noise in TopKGate to True because vitmoe_loss is set to True"
            )
            self.use_noise = True
        self.batch_prioritized_routing = kwargs.get("batch_prioritized_routing")
        if int(os.environ.get("BATCH_PRIO", 0)) != 0:
            self.batch_prioritized_routing = True
        self.use_global_loss = kwargs.get("use_global_loss")
        self.is_postscore = kwargs.get("is_postscore")

    def make_route_decision(self, score, score_wo_noise, gates):
        topk_logits, topk_indices = torch.topk(score, self.top_k, dim=1)

        indices_s = [x.view(-1) for x in topk_indices.chunk(self.top_k, dim=1)]

        masks_se = [
            one_hot_with_dtype(x, num_classes=self.num_global_experts, dtype=x.dtype)
            for x in indices_s
        ]

        gates_s = [(gates * x).sum(dim=1) for x in masks_se]

        with torch.cuda.amp.autocast(enabled=False):
            if self.num_global_experts <= 1:
                l_loss = torch.sum(score_wo_noise) * 0.0
            elif self.vitmoe_loss:
                gates_wo_noise = F.softmax(score_wo_noise, dim=1)
                l_loss = vitmoe_loss(
                    gates_wo_noise,
                    topk_logits,
                    self.num_global_experts,
                    self.use_global_loss,
                )
            else:
                l_loss = load_balance(
                    gates, masks_se[0], self.num_global_experts, self.fp32_gate
                )
        importance = -1 * gates.max(dim=1)[0]
        self.compute_location = get_compute_location_func(self.batch_prioritized_routing, importance)
        
        locations1 = self.compute_location(masks_se[0])

        locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]

        if self.top_k > 1:
            acc_base = None

            for k in range(1, self.top_k):
                acc_base = (
                    torch.sum(masks_se[k - 1], dim=0, keepdim=True)
                    if acc_base is None
                    else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True)
                )
                locations2 = self.compute_location(masks_se[k])
                locations2 += acc_base
                locations_s.append(
                    torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32)
                )

            # Normalize Gate
            if self.normalize_gate:
                denom_s = torch.clamp(
                    sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps
                )
                gates_s = [x / denom_s for x in gates_s]

        indices_s = [x.to(torch.int32) for x in indices_s]
        
        capacity = self.top_k * int(
                self.capacity_factor
                * ((score.size(0) + self.num_global_experts - 1) // self.num_global_experts)
            )
        
            

    def generate_hot_mask(self, score):

        pass

    def generate_auxiliary(self, score, gates):
        pass
