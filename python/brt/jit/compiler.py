# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

import torch
from brt._C import jit

__all__ = ["CUDACompiler"]


class CUDACompiler:
    @staticmethod
    def create_raw(source) -> Callable[..., None]:
        torch.cuda.init()
        kernel_type, __ctx__ = jit.inject_source(source)
        fo = open("foo.txt", "w")
        fo.write(source)
        fo.write("......................................")
        fo.write(kernel_type)
        fo.write("......................................")
        fo.write(str(__ctx__))
        fo.close()
        # import pdb; pdb.set_trace()
        if kernel_type == "global" or kernel_type == "horiz_fuse":

            def func(*inputs, extra=[]):
                jit.static_invoke(inputs, extra, __ctx__)

        elif kernel_type == "hetero_fuse":

            def func(*inputs, active_blocks=[]):
                jit.hetero_invoke(inputs, active_blocks, __ctx__)

        elif kernel_type == "homo_fuse":

            def func(shared_inputs, standalone_inputs, capacities=[]):
                jit.homo_invoke(shared_inputs, standalone_inputs, capacities, __ctx__)

        else:
            raise NotImplementedError
        return func

    @staticmethod
    def generate_kernel(keyword_dict, template: str) -> Callable[..., None]:
        if keyword_dict is not None:
            for key in keyword_dict:
                template = template.replace("@%s@" % key, str(keyword_dict[key]))
        return CUDACompiler.create_raw(template)
