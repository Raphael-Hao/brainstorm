#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /recur_module.py
# \brief:
# Author: raphael hao

# %%
import torch.nn as nn
import torch
from brainstorm import Router
from torch.jit import ScriptModule, script_method

#%%
class BranchModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scatter_router = Router()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x):
        x = x - 1
        x, y = self.scatter_router(x)
        x = self.linear1(x)
        y = self.linear2(y)
        return x


branch_net = BranchModule()
script_branch_net = torch.jit.script(branch_net)
script_branch_net_graph = script_branch_net.graph

print(script_branch_net_graph)


#%%
print("printing inputs of the graph")
for input in script_branch_net_graph.inputs():
    print(input)
print("printing nodes of the graph")
for node in script_branch_net_graph.nodes():
    print(node)
print("printing outputs of the graph")
for output in script_branch_net_graph.outputs():
    print(output)


# # %%
# import torch.nn as nn
# import torch


# class loop_model(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, x, y: int):
#         for i in range(y):  # for loop
#             x = self.linear(x)
#         return x


# script_loop_model = torch.jit.script(loop_model())

# print(script_loop_model.inlined_graph)

# # %%
# x = torch.Tensor([1])
# y = 10
# x = script_loop_model(x, y)
# print(x)
# torch.onnx.export(
#     script_loop_model, args=(x, y), f="script_loop_model.onnx", example_outputs=x
# )

# # %%

# import onnx

# onnx_script_loop_model = onnx.load("script_loop_model.onnx")
# onnx.checker.check_model(onnx_script_loop_model)

# #%%
# import onnxruntime as ort
# import numpy as np

# ort_session = ort.InferenceSession(
#     "script_loop_model.onnx", providers=["CUDAExecutionProvider"]
# )
# x = np.array([1], dtype=np.float32)
# y = np.array([10], dtype=np.int64)
# z = ort_session.run(None, {"x.1": x, "y.1": y})
# print(z[0])
# # %%
