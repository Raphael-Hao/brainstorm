# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
import torch

from brt.primitive import router
from .gather import GatherRouter
from .scatter import ScatterRouter
from tutel.gates.cosine_top import CosineTopKGate
from tutel.gates.top import LinearTopKGate


@router
class MoEInferenceScatterRouter(ScatterRouter):
    def __init__(
        self,
        global_expert,
        gating_func,
        topk=1,
        post_score=True,
    ):
        if isinstance(gating_func, nn.Module):
            route_func = gating_func
        elif isinstance(gating_func, str):
            if gating_func == "cosine":
                route_func = CosineTopKGate(topk)
            elif gating_func == "linear":
                route_func = LinearTopKGate(topk)
        else:
            raise ValueError("gating_func must be a nn.Module or a string")
        super().__init__(
            dst_num=global_expert,
            route_func=route_func,
            route_method="topk",
            transform=post_score,
        )


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
