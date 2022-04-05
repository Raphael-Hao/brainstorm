#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /recur_module.py
# \brief:
# Author: raphael hao

# %%
import torch.nn as nn
import torch
from brainstorm import Router, BranchPod
from torch.jit import ScriptModule, script_method

class recur_module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pod = BranchPod([lambda x: x, self])
        self.router = Router()

    def forward(self, x):
        x = x - 1
        self.router.route(x)
        x = self.router.scatter(x)
        x = self.pod(x)
        return x


mod = recur_module()

x = torch.Tensor([-1])
y = mod(x)
print(y)


# %%
import torch.nn as nn
import torch


class loop_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, y: int):
        for i in range(y):
            x = self.linear(x)
        return x


script_loop_model = torch.jit.script(loop_model())

print(script_loop_model.code)

# %%
x = torch.Tensor([1])
y = 10
x = script_loop_model(x, y)
print(x)
torch.onnx.export(
    script_loop_model, args=(x, y), f="script_loop_model.onnx", example_outputs=x
)

# %%

import onnx

onnx_script_loop_model = onnx.load("script_loop_model.onnx")
onnx.checker.check_model(onnx_script_loop_model)

#%%
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession(
    "script_loop_model.onnx", providers=["CUDAExecutionProvider"]
)
x = np.array([1], dtype=np.float32)
y = np.array([10], dtype=np.int64)
z = ort_session.run(None, {"x.1": x, "y.1": y})
print(z[0])
# %%
