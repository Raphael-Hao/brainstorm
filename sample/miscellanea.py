# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
#%%
# import brt
# import brt.nn as nn
# from brt.common import log
# from brt.frontend import build_graph
# from brt.router import RandomGatherRouter, RandomScatterRouter

# log.set_level("frontend", "DEBUG")

# @brt.netlet
# class MoE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.scatter_router = RandomScatterRouter(route_num=2)
#         self.expert1 = nn.Linear(10, 10)
#         self.expert2 = nn.Linear(10, 10)
#         self.gather_router = RandomGatherRouter(route_num=2)

#     def forward(self, x):
#         route_results, reverse_indice, reverse_shape = self.scatter_router(x)
#         x_0 = self.expert1(route_results[0])
#         x_1 = self.expert2(route_results[1])
#         x = self.gather_router([x_0, x_1], reverse_indice, reverse_shape)
#         return x


# @brt.domain
# class MoEModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.moe = MoE()

#     def forward(self, x):
#         return self.moe(x)


# moe_model = MoEModel()

# model_ir = build_graph(moe_model)

# %%

import onnx
from brt.runtime.jit.tvm import TVMTuner

tuner = TVMTuner()
model_name = "sparse_fusion_2_thor_model"
onnx_model = onnx.load(f"/home/whcui/brainstorm_project/brainstorm/log/{model_name}.onnx")
tuner.import_onnx_netlet(onnx_model, model_name)

# %%
