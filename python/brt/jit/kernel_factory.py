# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, List

import torch

from brt.runtime import BRT_KERNEL_TEMPLATE_PATH
from brt.jit.modules import ModuleInfo
from brt.jit.compiler import CUDACompiler
from brt.jit.codegen import (
    ModuleKernel,
    HorizFusedKernel,
    HeteroFusedKernel,
    HomoFusedKernel,
)

__all__ = ["make_jit_kernel"]


def make_jit_kernel(
    modules,
    sample_inputs,
    method="forward",
    opt_level=None,
    objective_func: str = "fastest",
    rank: int = 1,
) -> Callable[..., None]:
    if opt_level is None:
        kernel = ModuleKernelFactory.make_kernel(
            modules, method, sample_inputs, objective_func, rank
        )

    elif opt_level == "horiz_fuse":
        kernel = HorizFusedKernelFactory.make_kernel(
            modules, method, sample_inputs, objective_func, rank
        )

    elif opt_level == "hetero_fuse":
        kernel = HeteroFusedKernelFactory.make_kernel(
            modules, method, sample_inputs, objective_func, rank
        )

    elif opt_level == "homo_fuse":
        kernel = HomoFusedKernelFactory.make_kernel(
            modules, method, sample_inputs, objective_func, rank
        )
    else:
        raise ValueError(f"Not supported optimize level: {opt_level}")
    kernel_code, _, _, _ = kernel.get_code()

    processed_template_fname = str(
        BRT_KERNEL_TEMPLATE_PATH / ("processed_" + kernel.func_name[:10] + ".cu")
    )
    with open(processed_template_fname, "w") as f:
        f.write(kernel_code)

    jit_kernel = CUDACompiler.generate_kernel(None, kernel_code)
    return jit_kernel


class HorizFusedKernelFactory:
    @staticmethod
    def make_kernel(
        modules: torch.nn.ModuleList,
        method,
        sample_inputs,
        objective_func: str = "fastest",
        rank: int = 1,
    ):
        assert len(modules) == len(
            sample_inputs
        ), "modules and sample_inputs must have the same length"

        candidates = []
        for m, sample_input in zip(modules, sample_inputs):
            module_kernel = ModuleKernelFactory.make_kernel(
                m, method, sample_input, objective_func, rank
            )
            candidates.append(module_kernel)
        fused_kernel = HorizFusedKernel(candidates)
        return fused_kernel


class HeteroFusedKernelFactory:
    @staticmethod
    def make_kernel(
        modules: torch.nn.ModuleList,
        method,
        sample_inputs,
        objective_func: str = "fastest",
        rank: int = 1,
    ):
        assert len(modules) == len(
            sample_inputs
        ), "modules and sample_inputs must have the same length"

        candidates = []
        for m, sample_input in zip(modules, sample_inputs):
            module_kernel = ModuleKernelFactory.make_kernel(
                m, method, sample_input, objective_func, rank
            )
            candidates.append(module_kernel)
        fused_kernel = HeteroFusedKernel(candidates)
        return fused_kernel


class HomoFusedKernelFactory:
    @staticmethod
    def make_kernel(
        modules: torch.nn.ModuleList,
        method,
        sample_inputs: List[List[torch.Tensor]],
        objective_func: str = "fastest",
        rank: int = 1,
    ):
        HomoFusedKernelFactory.check_homogeneity(modules)
        candidate_module = modules[0]
        candidate_kernels = ModuleKernelFactory.make_kernels(
            candidate_module, method, sample_inputs, objective_func, rank
        )
        path_num = len(modules)
        capacities = [
            sample_input[0].size(0)
            if isinstance(sample_input, (list, tuple))
            else sample_input.size(0)
            for sample_input in sample_inputs
        ]
        shared_arg_indices = None
        shared_arg_grans = None
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(candidate_module):
                (
                    shared_arg_indices,
                    shared_arg_grans,
                ) = subclass.extract_shared_arg_infos(
                    candidate_module, method, sample_inputs[0]
                )
                break
        assert shared_arg_indices is not None, "shared_arg_indices is None"
        assert shared_arg_grans is not None, "shared_arg_grans is None"

        fused_kernel = HomoFusedKernel(
            path_num,
            capacities,
            shared_arg_indices,
            shared_arg_grans,
            candidate_kernels,
        )
        return fused_kernel

    @staticmethod
    def check_homogeneity(modules: torch.nn.ModuleList):
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


class ModuleKernelFactory:
    @staticmethod
    def make_kernel(
        module: torch.nn.Module,
        method,
        sample_input,
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> ModuleKernel:
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(module):
                return subclass.make_kernel(
                    module, method, sample_input, objective_func, rank
                )
        raise ValueError(f"Unknown module type: {module}")

    @staticmethod
    def make_kernels(
        module: torch.nn.Module,
        method,
        sample_inputs,
        objective_func: str = "fastest",
        rank: int = 1,
    ) -> List[ModuleKernel]:
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(module):
                ret_kernels = []
                for sample_input in sample_inputs:
                    ret_kernels.append(
                        subclass.make_kernel(
                            module, method, sample_input, objective_func, rank
                        )
                    )
                return ret_kernels
        raise ValueError(f"Unknown module type: {module}")
