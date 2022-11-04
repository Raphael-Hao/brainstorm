# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, List
from abc import ABC, abstractstaticmethod

import torch
from torch import nn

from brt.runtime import BRT_KERNEL_TEMPLATE_PATH
from brt.jit.modules import AtomModule
from brt.jit._function import make_jit_function


__all__ = ["make_jit_module"]


def make_jit_module(modules, sample_inputs=None, mode="eval", opt_level="none"):
    if mode == "eval":
        assert sample_inputs is not None, "sample_inputs must be provided in eval mode"

        jit_function = make_jit_function(modules, sample_inputs, mode, opt_level)

        if opt_level is None:
            assert len(modules) == 1, "only one module is supported for none opt mode"

    raise NotImplementedError("make_jit_module is not implemented yet.")


class AtomModuleFactory:
    @staticmethod
    def produce(
        module: nn.Module, sample_inputs=None, mode="eval", opt_level="none"
    ) -> nn.Module:
        self.make_


class HorizFusedModuleFactory:
    pass


class HeteroFusedModuleFactory:
    pass


class HomoFusedModuleFactory:
    pass
