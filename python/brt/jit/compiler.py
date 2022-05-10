# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import torch

from . import cppjit


class CUDACompiler:
    @staticmethod
    def create_raw(source):
        torch.cuda.init()
        kernel_type, __ctx__ = cppjit.inject_source(source)

        if kernel_type == "global" or kernel_type == "horiz_fuse":

            def func(*inputs, extra=[]):
                cppjit.static_invoke(inputs, extra, __ctx__)

        elif kernel_type == "hetero_fuse":

            def func(*inputs, active_blocks=[]):
                cppjit.hetero_invoke(inputs, active_blocks, __ctx__)

        elif kernel_type == "homo_fuse":

            def func(shared_inputs, weights, active_blocks=[]):
                cppjit.homo_invoke(shared_inputs, weights, active_blocks, __ctx__)

        elif kernel_type == "elastic_homo_fuse":
            raise NotImplementedError
        return func

    @staticmethod
    def generate_kernel(keyword_dict, template: str):
        if keyword_dict is not None:
            for key in keyword_dict:
                template = template.replace("@%s@" % key, str(keyword_dict[key]))
        return CUDACompiler.create_raw(template)

    @staticmethod
    def generate_hetero_kernel(keyword_dict, template: str):
        if keyword_dict is not None:
            for key in keyword_dict:
                template = template.replace("@%s@" % key, str(keyword_dict[key]))
        return CUDACompiler.create_raw(template)
