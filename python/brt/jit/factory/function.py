# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Union

import torch
from brt.routers.proto_tensor import collect_proto_attr_stack, init_proto_tensor

from .kernel import make_jit_kernel
from .registry import ModuleInfo

__all__ = ["make_jit_function"]

def make_jit_function(modules, sample_inputs=None, mode="eval", opt_level="none"):
    if mode == "eval":

        assert sample_inputs is not None, "sample_inputs must be provided in eval mode"

        jit_kernel = make_jit_kernel(modules, sample_inputs, "forward", opt_level)

        if opt_level == "none":
            assert len(modules) == 1, "only one module is supported for none opt mode"
            for subclass in ModuleInfo.__subclasses__():
                if subclass.ismodule(modules):
                    output_init_func = subclass.get_output_init_func(modules, "forward")
                    (
                        _,
                        _,
                        input_indices,
                        output_indices,
                    ) = subclass.extract_arg_infos(modules, "forward")
                break

            class JitFunction:
                @staticmethod
                def forward(*inputs):
                    inputs = list(inputs)
                    in_data = [inputs[i] for i in input_indices]
                    out_data = output_init_func(*in_data)
                    for i, out_index in enumerate(output_indices):
                        inputs.insert(out_index, out_data[i])
                    jit_kernel(*inputs)
                    outputs = [inputs[i] for i in output_indices]
                    return outputs

        elif opt_level == "hetero_fuse":
            assert isinstance(
                modules, torch.nn.ModuleList
            ), "modules must be a ModuleList for hetero fusion"
            input_arg_num_s = []
            total_arg_num_s = []
            input_indices_s = []
            output_indices_s = []
            output_init_func = []

            for m in modules:
                for subclass in ModuleInfo.__subclasses__():
                    if subclass.ismodule(m):
                        output_init_func.append(
                            subclass.get_output_init_func(m, "forward")
                        )
                        (
                            input_arg_num,
                            total_arg_num,
                            input_indices,
                            output_indices,
                        ) = subclass.extract_arg_infos(m, "forward")
                        input_arg_num_s.append(input_arg_num)
                        total_arg_num_s.append(total_arg_num)
                        input_indices_s.append(input_indices)
                        output_indices_s.append(output_indices)
                        break

            input_prefix = 0
            total_prefix = 0
            arg_start_indices = []
            arg_end_indices = []
            final_output_indices = []
            dst_num = len(modules)
            for i in range(dst_num):
                arg_start_indices.append(input_prefix)
                input_prefix += input_arg_num_s[i]
                arg_end_indices.append(input_prefix)
                for out_index in output_indices_s[i]:
                    final_output_indices.append(out_index + total_prefix)
                total_prefix += total_arg_num_s[i]

            class JitFunction:
                @staticmethod
                def forward(*inputs, active_blocks):
                    origin_inputs = list(inputs)
                    new_inputs = []
                    for i in range(dst_num):
                        inputs = origin_inputs[
                            arg_start_indices[i] : arg_end_indices[i]
                        ]
                        in_data = [inputs[j] for j in input_indices_s[i]]
                        out_data = output_init_func[i](*in_data)
                        for j, out_index in enumerate(output_indices_s[i]):
                            inputs.insert(out_index, out_data[j])
                        new_inputs.extend(inputs)

                    jit_kernel(*new_inputs, active_blocks)

                    outputs = [new_inputs[j] for j in final_output_indices]

                    return outputs

        elif opt_level == "homo_fuse":
            candidate_module = modules[0]
            for subclass in ModuleInfo.__subclasses__():
                if subclass.ismodule(candidate_module):
                    output_init_func = subclass.get_output_init_func(
                        candidate_module, "forward"
                    )
                    (
                        input_arg_num,
                        total_arg_num,
                        input_indices,
                        output_indices,
                    ) = subclass.extract_arg_infos(candidate_module, "forward")
                    break

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
                    outputs = shared_inputs[in_data_num:]
                    return outputs

    elif mode == "train":
        raise NotImplementedError

        class JitFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                return jit_kernel.forward(inputs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                return jit_kernel.backward(grad_outputs)

    return JitFunction
