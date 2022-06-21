# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Union

import torch
import torch.nn.functional as F

from .kernel import JitKernel
from .registry import ModuleInfo


class FunctionFactory:
    @staticmethod
    def make_function(self, modules, jit_kernel, mode="eval", opt_mode="none"):
        if mode == "eval":
            if opt_mode == "none":
                assert (
                    len(modules) == 1
                ), "only one module is supported for none opt mode"
                for subclass in ModuleInfo.__subclasses__():
                    if subclass.ismodule(modules):
                        output_init_func = subclass.get_output_init_func(
                            modules, "forward"
                        )
            elif opt_mode == "hetero_fuse":
                output_init_func = []
                for m in modules:
                    for subclass in ModuleInfo.__subclasses__():
                        if subclass.ismodule(m):
                            output_init_func.append(
                                subclass.get_output_init_func(m, "forward")
                            )
            elif opt_mode == "homo_fuse":
                candidate_module = modules[0]
                for subclass in ModuleInfo.__subclasses__():
                    if subclass.ismodule(candidate_module):
                        output_init_func = subclass.get_output_init_func(
                            candidate_module, "forward"
                        )

            class JitFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, inputs):
                    return jit_kernel.forward(inputs)

        elif mode == "train":

            class JitFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, inputs):
                    return jit_kernel.forward(inputs)

                @staticmethod
                def backward(ctx, *grad_outputs):
                    return jit_kernel.backward(grad_outputs)

        return JitFunction
