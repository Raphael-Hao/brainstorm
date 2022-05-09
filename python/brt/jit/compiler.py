# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import torch

from . import cppjit


class CUDACompiler:
    @staticmethod
    def create_raw(source):
        torch.cuda.init()
        __ctx__ = cppjit.inject_source(source)

        def func(*inputs, extra=[]):
            cppjit.static_invoke(inputs, extra, __ctx__)

        return func

    @staticmethod
    def generate_kernel(keyword_dict, template: str):
        if keyword_dict is not None:
            for key in keyword_dict:
                template = template.replace("@%s@" % key, str(keyword_dict[key]))
        return CUDACompiler.create_raw(template)

