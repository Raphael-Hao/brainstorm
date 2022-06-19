# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict

import torch


class Operator(torch.autograd.Function):
    FUNC_POOL = defaultdict(list)

    @staticmethod
    def forward(ctx, jit_function, inputs):
        ctx.jit_function = jit_function
        return ctx.jit_function.forward(inputs)

    @staticmethod
    def backward(ctx, inputs):
        return ctx.jit_function.backward(inputs)
