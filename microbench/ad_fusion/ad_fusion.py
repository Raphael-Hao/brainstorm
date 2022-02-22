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
import numpy as np

#%%
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (64,), dtype="float32")
        B = T.match_buffer(b, (64,), dtype="float32")
        for i in range(64):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


ir_module = MyModule
print(type(ir_module))
print(ir_module.script())

#%%
mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
print(type(mod))

a = tvm.nd.array(np.arange(64).astype("float32"))
b = tvm.nd.array(np.zeros((64,)).astype("float32"))
mod(a, b)
print(a)
print(b)

#%%
sch = tvm.tir.Schedule(ir_module)
print(type(sch))
# Get block by its name
block_b = sch.get_block("B")
# Get loops surronding the block
(i,) = sch.get_loops(block_b)
# Tile the loop nesting.
i_0, i_1, i_2 = sch.split(i, factors=[2, 4, 8])
print(sch.mod.script())

#%%

################################################################################################
# We can also reorder the loops. Now we move loop `i_2` to outside of `i_1`.
sch.reorder(i_0, i_2, i_1)
print(sch.mod.script())


################################################################################################
# Transform to a GPU program
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# If we want to deploy models on GPUs, thread binding is necessary. Fortunately, we can
# also use primitives and do incrementally transformation.
#

sch.bind(i_0, "blockIdx.x")
sch.bind(i_2, "threadIdx.x")
print(sch.mod.script())


################################################################################################
# After binding the threads, now build the IRModule with :code:`cuda` backends.
ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")
cuda_a = tvm.nd.array(np.arange(64).astype("float32"), ctx)
cuda_b = tvm.nd.array(np.zeros((64,)).astype("float32"), ctx)
cuda_mod(cuda_a, cuda_b)
print(cuda_a)
print(cuda_b)

# %%
