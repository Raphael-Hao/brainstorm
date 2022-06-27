# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from .function import make_jit_function
from .registry import ModuleInfo


def make_jit_module(modules, sample_inputs=None, mode="eval", opt_level="none"):
    if mode == "eval":
        assert sample_inputs is not None, "sample_inputs must be provided in eval mode"
        
        jit_function = make_jit_function(modules, sample_inputs, mode, opt_level)
        
        if opt_level is None:
            assert len(modules) == 1, "only one module is supported for none opt mode"
