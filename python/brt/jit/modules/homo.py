import itertools
from typing import List, Tuple, Union, Literal, Callable

import torch
from torch import autograd
from torch import nn

from brt.jit.codegen.homo_fused import HomoFusedKernel
from brt.jit.modules.base import FuseModuleInputType
from brt.jit.modules.fused import FusedModule


class HomoFusedModule(FusedModule):
    def __init__(self, module: nn.ModuleList):
        super().__init__(module)
        self._check_homogeneity(self.module)

    def _make_global_kernel(
        self,
        sample_inputs: FuseModuleInputType,
        method: str = "forward",
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> HomoFusedKernel:
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

        candidates = []
        if isinstance(rank, int):
            rank = [rank] * len(sample_inputs)
        else:
            assert len(rank) == len(sample_inputs)
        for idx, inp in enumerate(sample_inputs):
            module_kernel = self.jit_submodules[0]._make_global_kernel(
                inp, method, objective_func, rank[idx]
            )
            candidates.append(module_kernel)
        fused_kernel = HomoFusedKernel(
            self.num_submodule,
            capacities,
            shared_arg_indices,
            shared_arg_grans,
            candidates,
        )
        return fused_kernel

    def make_function(
        self,
        sample_inputs: FuseModuleInputType,
        mode: Literal["eval", "train"] = "eval",
        objective_func: str = "fastest",
        rank: Union[int, List[int]] = 1,
    ) -> autograd.Function:
        raise NotImplementedError
        jit_kernel = self.make_kernel(
            sample_inputs=sample_inputs,
            method="forward",
            objective_func=objective_func,
            rank=rank,
        )

        candidate = self.jit_submodules[0]
        output_shape_dict = {
            inp.shape[0]: candidate._get_output_shape("forward", inp)
            for inp in sample_inputs
        }
        # for subclass in AtomModule.__subclasses__():
        #     if subclass.ismodule(candidate_module):
        #         output_init_func = subclass.get_output_init_func(
        #             candidate_module, "forward"
        #         )
        (
            input_arg_num,
            total_arg_num,
            input_indices,
            output_indices,
        ) = candidate._extract_arg_infos("forward")
        out_data = [
            torch.empty(shp).to("cuda")
            for shp in self._get_output_shape("forward", sample_inputs)
        ]

        in_data_num = len(input_indices)
        out_data_num = len(output_indices)

        class JitFunction:
            @staticmethod
            def forward(*inputs, capacities):
                inputs = list(inputs)
                in_data = inputs[:in_data_num]
                standalone_inputs = inputs[in_data_num:]
                out_data = output_init_func(*in_data)
                shared_inputs = in_data + out_data
                jit_kernel(shared_inputs, standalone_inputs, capacities)
                start_idx = in_data_num
                outputs = []
                for cap in capacities:
                    outputs.append(shared_inputs[start_idx : start_idx + cap])
                    start_idx += cap
                return tuple(outputs)

        return JitFunction

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
        sample_inputs: FuseModuleInputType,
    ) -> Tuple[List, List]:
        raise NotImplementedError()

    def _extract_arg_infos(
        self,
        method: str,
    ) -> Tuple[int, int, List[int], List[int]]:
        raise NotImplementedError()

    def _get_output_shape(
        self, method: str, sample_inputs: FuseModuleInputType
    ) -> List[torch.Size]:
        candidate = self.jit_submodules[0]
        return list(
            itertools.chain.from_iterable(
                candidate._get_output_shape(method, inp) for inp in sample_inputs
            )
        )

    @property
    def module_name(self) -> str:
        return f"HomoFused_{self.num_submodule}_" + "_".join(
            [jsm.module_name for jsm in self.jit_submodules]
        )
