# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
import re
from typing import Tuple

import tvm


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


def get_culaunch_config(tvm_ir: tvm.IRModule) -> str:
    grid_dim, block_dim = parse_culaunch_config(tvm_ir)
    culaunch_config = f"// [thread_extent] blockIdx.xdim = {grid_dim[0]}\n"
    culaunch_config += f"// [thread_extent] blockIdx.ydim = {grid_dim[1]}\n"
    culaunch_config += f"// [thread_extent] blockIdx.zdim = {grid_dim[2]}\n"
    culaunch_config += f"// [thread_extent] threadIdx.xdim = {block_dim[0]}\n"
    culaunch_config += f"// [thread_extent] threadIdx.ydim = {block_dim[1]}\n"
    culaunch_config += f"// [thread_extent] threadIdx.zdim = {block_dim[2]}\n"
    return culaunch_config
