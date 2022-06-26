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
    op_type: str,
    method: str,
    input_infos: Dict[str, List[int]],
    output_infos: Dict[str, List[int]],
    parameters: Dict[str, Union[Union[int, float], List[Union[int, float]]]],
) -> str:
    fname = "_".join([op_type, method])
    fname += "_"
    fname += "-".join(
        f"{name}_" + "_".join(str(dim) for dim in shape)
        for name, shape in input_infos.items()
    )
    fname += "_"
    fname += "_".join(
        f"{name}_" + "_".join(str(dim) for dim in shape)
        for name, shape in output_infos.items()
    )
    fname += "_"
    fname += "_".join(
        f"{name}_" + "_".join(str(dim) for dim in parameter)
        if isinstance(parameter, (list, tuple))
        else f"{name}_" + str(parameter)
        for name, parameter in parameters.items()
    )
    return fname


def old_make_fname(
    op_type,
    input_infos: Dict[str, List[int]],
    output_infos: Dict[str, List[int]],
    parameters: Dict[str, Union[Union[int, float], List[Union[int, float]]]],
) -> str:
    fname = op_type
    fname += "--"
    fname += "-".join(
        f"{name}-" + "-".join(str(dim) for dim in shape)
        for name, shape in input_infos.items()
    )
    fname += "--"
    fname += "-".join(
        f"{name}-" + "-".join(str(dim) for dim in shape)
        for name, shape in output_infos.items()
    )
    fname += "--"
    fname += "-".join(
        f"{name}-" + "-".join(str(dim) for dim in parameter)
        if isinstance(parameter, list)
        else f"{name}-" + str(parameter)
        for name, parameter in parameters.items()
    )
    return fname
