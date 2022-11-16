from typing import List, Tuple, Union, Literal, Callable

import torch

from brt.jit.codegen.homo_fused import HomoFusedKernel
from brt.jit.modules.fused import FusedModule


class HomoFusedModule(FusedModule):
    def _make_global_kernel(
        self,
        sample_inputs: List[torch.Tensor],
        method: str = "forward",
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> HomoFusedKernel:
        self._check_homogeneity(self.module)
        capacities = [
            sample_input[0].size(0)
            if isinstance(sample_input, (list, tuple))
            else sample_input.size(0)
            for sample_input in sample_inputs
        ]
        shared_arg_indices = None
        shared_arg_grans = None
        (shared_arg_indices, shared_arg_grans,) = self.jit_submodules[
            0
        ]._extract_shared_arg_infos(method, sample_inputs[0])
        assert shared_arg_indices is not None, "shared_arg_indices is None"
        assert shared_arg_grans is not None, "shared_arg_grans is None"

        candidates = [jsm._make_global_kernel for jsm in self.jit_submodules]
        fused_kernel = HomoFusedKernel(
            self.num_submodule,
            capacities,
            shared_arg_indices,
            shared_arg_grans,
            candidates,
        )
        return fused_kernel

    @staticmethod
    def _check_homogeneity(modules: torch.nn.ModuleList):
        module_class_name = type(modules[0]).__name__
        """ TODO
        Currently we only check class name.
        We should check the attributes
        """
        for m in modules:
            if type(m).__name__ != module_class_name:
                raise ValueError(
                    "modules must be homogeneous. "
                    "Found {} and {}".format(module_class_name, type(m).__name__)
                )

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
        return f"HomoFused_{self.num_submodule}_" + "_".join(
            [jsm.module_name for jsm in self.jit_submodules]
        )
