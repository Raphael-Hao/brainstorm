# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import re
from typing import Dict, List, Tuple, Union

import torch

import tvm

__all__ = [
    "parse_culaunch_config",
    "make_culaunch_config_str",
    "make_inputs",
    "make_fname",
]


def parse_culaunch_config(
    tvm_ir: tvm.IRModule,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    tvm_ir_str = str(tvm_ir)
    rule = 'attr \[IterVar\((\w+\.\w+): int32, \(nullptr\), "ThreadIndex", "(\w+\.\w+)"\)\] "thread_extent" = (\d+)'
    res = re.findall(rule, tvm_ir_str)
    size = {
        "blockIdx.x": 1,
        "blockIdx.y": 1,
        "blockIdx.z": 1,
        "threadIdx.x": 1,
        "threadIdx.y": 1,
        "threadIdx.z": 1,
    }
    for r in res:
        if r[0] == r[1]:
            size[r[0]] = int(r[2])
    return (
        (size["blockIdx.x"], size["blockIdx.y"], size["blockIdx.z"]),
        (size["threadIdx.x"], size["threadIdx.y"], size["threadIdx.z"]),
    )


def make_culaunch_config_str(grid_dim, block_dim) -> str:
    culaunch_config = f"// [thread_extent] blockIdx.x = {grid_dim[0]}\n"
    culaunch_config += f"// [thread_extent] blockIdx.y = {grid_dim[1]}\n"
    culaunch_config += f"// [thread_extent] blockIdx.z = {grid_dim[2]}\n"
    culaunch_config += f"// [thread_extent] threadIdx.x = {block_dim[0]}\n"
    culaunch_config += f"// [thread_extent] threadIdx.y = {block_dim[1]}\n"
    culaunch_config += f"// [thread_extent] threadIdx.z = {block_dim[2]}\n"
    return culaunch_config


def make_inputs(
    input_infos: Dict[str, List[int]], input_dtype=None
) -> List[torch.Tensor]:
    inputs = []
    for input_name, input_shape in input_infos.items():
        inputs.append(torch.randn(input_shape, dtype=input_dtype))
    return inputs


def make_fname(
    op_type,
    input_infos: Dict[str, List[int]],
    output_infos: Dict[str, List[int]],
    parameters: Dict[str, List[Union[int, float]]],
) -> str:
    fname = op_type
    fname += "_\{"
    fname += "_".join(
        "\[" + "_".join(str(dim) for dim in shape) + "\]"
        for shape in input_infos.values()
    )
    fname += "\}_\{"
    fname += "_".join(
        "\[" + "_".join(str(dim) for dim in shape) + "\]"
        for shape in output_infos.values()
    )
    fname += "\}_\{"
    fname += "_".join(
        "\[" + "_".join(str(dim) for dim in parameter) + "\]"
        if isinstance(parameter, list)
        else str(parameter)
        for parameter in parameters.values()
    )
    fname += "\}"
    return fname
