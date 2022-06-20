# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch


class JitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, jit_kernel, inputs):
        ctx.jit_kernel = jit_kernel
        return ctx.jit_function.forward(inputs)

    @staticmethod
    def backward(ctx, inputs):
        return ctx.jit_function.backward(inputs)
