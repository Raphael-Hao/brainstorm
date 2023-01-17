# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Tuple, Union

import torch

from brt.runtime import log
from brt.jit.modules.atom import AtomModule, AtomModuleInputType
from brt.jit.codegen.module import ModuleKernel, ModuleDTypeSizeInByte

logger = log.get_logger(__file__)


class LinearModule(AtomModule):

    _input_arg_indices = {"forward": [0]}
    _output_arg_indices = {"forward": [2]}
    _shared_arg_indices = {"forward": [0, 2]}

    _involved_module_cls = torch.nn.Linear

    @classmethod
    def ismodule(cls, module: torch.nn.Module) -> bool:
        return isinstance(module, cls._involved_module_cls)

    def _make_global_kernel(
        self,
        sample_inputs: AtomModuleInputType,
        method: str,
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> ModuleKernel:
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        sample_output = self.module(sample_inputs)
        input_infos = {"input_0": list(sample_inputs.shape)}
        output_infos = {"output_0": list(sample_output.shape)}
        parameters = {
            "in_features": self.module.in_features,
            "out_features": self.module.out_features,
        }
        return ModuleKernel(
            module_name=self.module_name,
            method=method,
            kernel_source=None,
            input_infos=input_infos,
            output_infos=output_infos,
            parameters=parameters,
        ).load_from_db(objective_func, rank)

    def _extract_shared_arg_infos(
        self, method: str, sample_input: torch.Tensor
    ) -> Tuple[List, List]:
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        shared_arg_grans = [
            self.module.in_features * ModuleDTypeSizeInByte[sample_input.dtype],
            self.module.out_features * ModuleDTypeSizeInByte[sample_input.dtype],
        ]
        return type(self)._shared_arg_indices[method], shared_arg_grans

    def _extract_arg_infos(self, method: str) -> Tuple[int, int, List[int], List[int]]:
        if method not in type(self)._shared_arg_indices:
            raise NotImplementedError(f"{method} is not supported")
        if self.module.bias is None:
            input_arg_num = 2
        else:
            input_arg_num = 3
        total_arg_num = input_arg_num + 1
        return (
            input_arg_num,
            total_arg_num,
            type(self)._input_arg_indices[method],
            type(self)._output_arg_indices[method],
        )

    @property
    def module_name(self) -> str:
        return "Linear" if self.module.bias is None else "LinearBias"
