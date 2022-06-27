#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /ad_fusion.py
# \brief:
# Author: raphael hao

#%%
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm import meta_schedule

import numpy as np

#%%
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (64, 64), dtype="float32")
        B = T.match_buffer(b, (64, 64), dtype="float32")
        C = T.match_buffer(c, (64, 64), dtype="float32")

        for yo, xo, ko in T.grid(16, 16, 16):
            with T.block("B"):
                vy = T.axis.spatial(16, value=yo)
                vx = T.axis.spatial(16, value=xo)
                vk = T.axis.reduce(16, value=ko)
                with T.init():
                    for yi, xi, ki in T.grid(4, 4, 4):
                        C[vy * 4 + yi, vx * 4 + xi] = 0.0
                for yi, xi, ki in T.grid(4, 4, 4):
                    C[vy * 4 + yi, vx * 4 + xi] += (
                        A[vy * 4 + yi, vk * 4 + ki] * B[vk * 4 + ki, vx * 4 + xi]
                    )


ir_module = MyModule
print(ir_module.script())

#%%
mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.

a = tvm.nd.array(np.ones((64, 64)).astype("float32"))
b = tvm.nd.array(np.ones((64, 64)).astype("float32"))
c = tvm.nd.array(np.ones((64, 64)).astype("float32"))
mod(a, b, c)
print(a)
print(b)
print(c)

#%%
sch = tvm.tir.Schedule(ir_module)
# Get block by its name
block_b = sch.get_block("B")
# Get loops surronding the block
i, j, k = sch.get_loops(block_b)

print(sch.mod.script())

# %%
################################################################################################
# Transform to a GPU program
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# If we want to deploy models on GPUs, thread binding is necessary. Fortunately, we can
# also use primitives and do incrementally transformation.
#

sch.bind(i, "blockIdx.x")
sch.bind(j, "blockIdx.y")
print(sch.mod.script())

# %%

################################################################################################
# After binding the threads, now build the IRModule with :code:`cuda` backends.
ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
cuda_source = cuda_mod.imported_modules[0].get_source()
print(cuda_source)

# %%
cuda_a = tvm.nd.array(np.ones((64, 64)).astype("float32"), ctx)
cuda_b = tvm.nd.array(np.ones((64, 64)).astype("float32"), ctx)
cuda_c = tvm.nd.array(np.ones((64, 64)).astype("float32"), ctx)
cuda_mod(cuda_a, cuda_b, cuda_c)
print(cuda_a)
print(cuda_b)
print(cuda_c)
# %%
