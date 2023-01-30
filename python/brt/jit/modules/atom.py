# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import List, Tuple, Union, Literal, Any, Optional, Type, Dict

import torch
from torch import autograd
from torch import nn
from torch.overrides import handle_torch_function, wrap_torch_function, has_torch_function

from brt.runtime import log
from brt.jit.modules.base import ModuleBase, JitModuleBase, AtomModuleInputType
from brt.jit.codegen.module import ModuleKernel
from brt.runtime import ProtoTensor

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
        sample_inputs: AtomModuleInputType,
        mode: Literal["eval", "train"] = "eval",
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> autograd.Function:
        if mode == "eval":
            jit_kernel = self.make_kernel(
                sample_inputs, "forward", objective_func, rank
            )
            # wrapped_jit_kernel = wrap_torch_function(lambda *x: x)(jit_kernel)
            (
                input_arg_num,
                total_arg_num,
                input_arg_indices,
                output_arg_indices,
            ) = self._extract_arg_infos("forward")
            out_data = [
                torch.empty(shp, device="cuda")
                for shp in self._get_output_shape("forward", sample_inputs)
            ]

            class JitFunction(autograd.Function):
                @staticmethod
                @wrap_torch_function(lambda *x: x)
                def forward(ctx: Any, *inputs):
                    inputs = list(inputs)
                    for i, out_index in enumerate(output_arg_indices):
                        inputs.insert(out_index, out_data[i])
                    # init proto tensor
                    jit_kernel(*inputs)
                    outputs = [inputs[i] for i in output_arg_indices]
                    return tuple(outputs)

                @staticmethod
                def backward(ctx: Any, *grad_outputs: Any) -> Any:
                    raise NotImplementedError

            return JitFunction

        elif mode == "train":
            raise NotImplementedError
        else:
            raise ValueError

    def _get_output_shape(
        self, method: str, sample_inputs: AtomModuleInputType
    ) -> List[torch.Size]:
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        if isinstance(sample_inputs, (list, tuple)):
            outputs = self.module.__getattribute__(method)(*sample_inputs)
        else:
            outputs = self.module.__getattribute__(method)(sample_inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return [o.shape for o in outputs]

    # @abstractmethod
    def _extract_parameters_and_buffers(self) -> List[Optional[torch.Tensor]]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def ismodule(cls, module: nn.Module) -> bool:
        raise NotImplementedError()

class JitAtomModule(JitModuleBase):
    def __init__(
        self,
        function: Type[autograd.Function],
        module_name: str = "BRT.AtomModule",
        extra_repr: str = "",
        parameters: Dict[str, torch.Tensor] = {},
    ):
        super().__init__(function, module_name, extra_repr, parameters)
        self._factory_cls = AtomModule
    
    @abstractmethod
    def forward(self):
        pass