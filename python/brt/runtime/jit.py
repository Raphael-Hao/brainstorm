# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.runtime import cppjit


class JitCompiler:
    @staticmethod
    def create_raw(source):
        torch.cuda.init()
        __ctx__ = cppjit.inject_source(source)

        def func(*inputs, extra=[]):
            cppjit.invoke(inputs, extra, __ctx__)

        return func

    @staticmethod
    def generate_kernel(keyword_dict, template: str):
        for key in keyword_dict:
            template = template.replace("@%s@" % key, str(keyword_dict[key]))
        return JitCompiler.create_raw(template)
