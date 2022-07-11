# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from brt.routers import GatherRouter, ScatterRouter
from brt.routers.proto_tensor import (
    ProtoTensor,
    init_proto_tensor,
    make_proto_tensor_cls,
    to_proto_tensor,
    to_torch_tensor,
)
from tutel.gates.cosine_top import CosineTopKGate
from tutel.gates.top import LinearTopKGate


class CosineRouteFunc(nn.Module):
    def __init__(self, model_dim, global_expert_num, **kwargs):
        super().__init__()
        self.capacity_factor = kwargs.pop("capacity_factor", 1)
        self.gate_noise = kwargs.pop("gate_noise")
        k = kwargs.pop("k", 1)
        fp32_gate = kwargs.pop("fp32_gate", False)
        proj_dim = kwargs.pop("proj_dim", 256)
        int_t = kwargs.pop("int_t", 0.5)
        self.global_expert_num = global_expert_num
        self.cosine_gate = CosineTopKGate(
            model_dim, global_expert_num, k, fp32_gate, proj_dim, int_t
        )

    def forward(self, x):
        logits = self.cosine_gate(x)
        if self.training and self.gate_noise > 0:
            logits_w_noise = (
                logits
                + self.gate_noise * torch.randn_like(logits) / self.global_expert_num
            )
        else:
            logits_w_noise = logits
        scores = F.softmax(logits_w_noise, dim=1)


class MoEInferenceScatterRouter(ScatterRouter):
    def __init__(
        self,
        global_expert_num,
        model_dim,
        gating_func,
        topk=1,
        fp32_gate=False,
        post_score=True,
    ):
        route_func = None
        if isinstance(gating_func, nn.Module):
            route_func = gating_func
        elif isinstance(gating_func, str):
            if gating_func.lower() == "cosine":
                route_func = CosineTopKGate(
                    model_dim=model_dim,
                    num_global_experts=global_expert_num,
                    k=topk,
                    fp32_gate=fp32_gate,
                )
            elif gating_func.lower() == "linear":
                route_func = LinearTopKGate(
                    model_dim=model_dim,
                    num_global_experts=global_expert_num,
                    k=topk,
                    fp32_gate=fp32_gate,
                )
        else:
            raise ValueError("gating_func must be a nn.Module or a string")
        assert route_func is not None
        super().__init__(
            path_num=global_expert_num,
            route_func=route_func,
            route_method="topk",
            transform=post_score,
        )

    def route(self, in_flow: Union[torch.Tensor, ProtoTensor]) -> List[ProtoTensor]:
        self.pack_invalid_flow(in_flow)
        in_flow_data, in_flow_tags, in_flow_loads, _ = to_torch_tensor(
            in_flow, copy_stack=True
        )


class MoeInferenceGatherRouter(GatherRouter):
    def __init__(self, dst_num: int, reduction: str = "add", sparse=True):
        super().__init__(dst_num, reduction, sparse)


# @router
# class MoETrainingScatterRouter(ScatterRouter):
#     def __init__(
#         self,
#         global_expert,
#         gating_func,
#         topk=1,
#         post_score=False,
#     ):
#         super().__init__(
#             dst_num=global_expert, route_func, route_method, residual_dst, transform, **kwargs
#         )
