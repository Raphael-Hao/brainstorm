# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List, Tuple, Union, Literal

import torch
from torch import autograd
from torch import nn

from brt.runtime import log
from brt.jit.modules.base import ModuleBase
from brt.jit.codegen.module import ModuleKernel

logger = log.get_logger(__file__)

__all__ = ["AtomModule"]


class AtomModule(ModuleBase):
    _input_arg_indices = None
    _output_arg_indices = None
    _shared_arg_indices = None

    def __init__(self, module: nn.Module):
        assert type(self).ismodule(module)
        super().__init__(module)

    def make_function(
        self,
        sample_inputs: Union[torch.Tensor, List[torch.Tensor]],
        mode: Literal["eval", "train"] = "eval",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> autograd.Function:
        if mode == "eval":
            jit_kernel = self.make_kernel(
                sample_inputs, "forward", objective_func, rank
            )
            (
                input_arg_num,
                total_arg_num,
                input_arg_indices,
                output_arg_indices,
            ) = self._extract_arg_infos("forward")
            out_data = [torch.empty(shp, device="cuda") for shp in self._get_output_shape("forward", sample_inputs)]

            class JitFunction(autograd.Function):
                @staticmethod
                def forward(*inputs):
                    inputs = list(inputs)
                    # in_data = [inputs[i] for i in input_arg_indices]
                    for i, out_index in enumerate(output_arg_indices):
                        inputs.insert(out_index, out_data[i])
                    jit_kernel(*inputs)
                    outputs = [inputs[i] for i in output_arg_indices]
                    return tuple(outputs)

            return JitFunction

        elif mode == "train":
            raise NotImplementedError
        else:
            raise ValueError

    @classmethod
    @abstractmethod
    def ismodule(cls, module: nn.Module) -> bool:
        raise NotImplementedError()
