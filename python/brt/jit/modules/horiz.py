from typing import List, Tuple, Union, Literal, Callable

import torch
from torch import nn
from torch import autograd

from brt.jit.codegen.horiz_fused import HorizFusedKernel
from brt.jit.modules.fused import FusedModule


class HorizFusedModule(FusedModule):
    def _make_global_kernel(
        self,
        sample_inputs: List[torch.Tensor],
        method: str = "forward",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> HorizFusedKernel:
        assert self.num_submodule == len(
            sample_inputs
        ), "modules and sample_inputs must have the same length"
        if isinstance(rank, int):
            rank = [rank] * self.num_submodule
        candidates = []
        for jsm, inp, rk in zip(self.jit_submodules, sample_inputs, rank):
            module_kernel = jsm._make_global_kernel(inp, method, objective_func, rk)
            candidates.append(module_kernel)
        fused_kernel = HorizFusedKernel(candidates)
        return fused_kernel

    def _extract_shared_arg_infos(
        self,
        method: str,
        sample_inputs: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[List, List]:
        raise NotImplementedError()

    def _extract_arg_infos(
        self,
        method: str,
    ) -> Tuple[int, int, List, List]:
        raise NotImplementedError()

    @property
    def module_name(self) -> str:
        return f"HorizFused_{self.num_submodule}_" + "_".join(
            [jsm.module_name for jsm in self.jit_submodules]
        )
