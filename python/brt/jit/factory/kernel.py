# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Dict, List, Union

import torch
import torch.nn.functional as F

from ..compiler import CUDACompiler
from ..kernel.hetero_fused import HeteroFusedKernel
from ..kernel.homo_fused import HomoFusedKernel
from ..kernel.module import ModuleKernel
from .registry import ModuleInfo

__all__ = [
    "JitKernel"
    "KernelFactory",
]


class JitKernel:
    @staticmethod
    def forward(*inputs):
        raise NotImplementedError

    @staticmethod
    def backward(*grad_outputs):
        raise NotImplementedError


class KernelFactory:
    @staticmethod
    def make_kernel(modules, sample_inputs, mode="eval", opt_mode="none"):
        jit_kernel = JitKernel()
        if opt_mode == "none":
            kernel = ModuleKernelFactory.make_kernel(modules, "forward", sample_inputs)
            kernel_code, _, _, _ = kernel.get_code()
            jit_kernel.forward = CUDACompiler.generate_kernel(None, kernel_code)

        elif opt_mode == "hetero_fuse":
            kernel = HeteroFusedKernelFactory.make_kernel(
                modules, "forward", sample_inputs
            )

        elif opt_mode == "homo_fuse":
            kernel = HomoFusedKernelFactory.make_kernel(
                modules, "forward", sample_inputs
            )
        kernel_code, _, _, _ = kernel.get_code()
        jit_kernel.forward = CUDACompiler.generate_kernel(None, kernel_code)
        if mode == "train":
            if opt_mode == "none":
                jit_kernel.backward = ModuleKernelFactory.make_kernel(
                    modules, "backward", sample_inputs
                )
            elif opt_mode == "hetero_fuse":
                raise ValueError("hetero_fuse is not supported in training mode")
                # jit_kernel.forward = HeteroFusedKernelFactory.make_kernel(
                #     modules, "backward", sample_inputs
                # )
            elif opt_mode == "homo_fuse":
                raise ValueError("homo_fuse is not supported in training mode")
                # jit_kernel.forward = HomoFusedKernelFactory.make_kernel(
                #     modules, "backward", sample_inputs
                # )
        return jit_kernel


class HeteroFusedKernelFactory:
    @staticmethod
    def make_kernel(modules: torch.nn.ModuleList, method, sample_inputs):
        candidates = []
        assert len(modules) == len(
            sample_inputs
        ), "modules and sample_inputs must have the same length"
        for m, sample_input in zip(modules, sample_inputs):
            module_kernel = ModuleKernelFactory.make_kernel(m, method, sample_input)
            candidates.append(module_kernel)
        fused_kernel = HeteroFusedKernel(candidates)
        return fused_kernel


class HomoFusedKernelFactory:
    @staticmethod
    def make_kernel(
        modules: torch.nn.ModuleList, method, sample_inputs: List[List[torch.Tensor]]
    ):
        HomoFusedKernelFactory.check_homogeneity(modules)
        candidate_module = modules[0]
        candidate_kernels = ModuleKernelFactory.make_kernels(
            candidate_module, method, sample_inputs
        )
        dst_num = len(modules)
        capacities = [sample_input[0].size[0] for sample_input in sample_inputs]
        shared_arg_indices = None
        shared_arg_grans = None
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(candidate_module):
                shared_arg_indices, shared_arg_grans = subclass.extract_argument_infos(
                    candidate_module, method, sample_inputs[0]
                )
                break
        assert shared_arg_indices is not None, "shared_arg_indices is None"
        assert shared_arg_grans is not None, "shared_arg_grans is None"

        fused_kernel = HomoFusedKernel(
            dst_num, capacities, shared_arg_indices, shared_arg_grans, candidate_kernels
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
    def make_kernel(module: torch.nn.Module, method, sample_input) -> ModuleKernel:
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(module):
                return subclass.make_kernel(module, method, sample_input)
        raise ValueError(f"Unknown module type: {module}")

    @staticmethod
    def make_kernels(
        module: torch.nn.Module, method, sample_inputs
    ) -> List[ModuleKernel]:
        for subclass in ModuleInfo.__subclasses__():
            if subclass.ismodule(module):
                ret_kernels = []
                for sample_input in sample_inputs:
                    ret_kernels.append(
                        subclass.make_kernel(module, method, sample_input)
                    )
                return ret_kernels
        raise ValueError(f"Unknown module type: {module}")
