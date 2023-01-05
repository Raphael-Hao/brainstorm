# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import os

os.environ["BRT_CACHE_PATH"] = "/home/v-weihaocui/brainstorm_project/brainstorm_raphael/.cache"

import importlib
import torch
import torch.nn as nn
from brt.jit.codegen.module import ModuleKernel
from brt.jit import make_jit_kernel
from brt.runtime.benchmark import CUDATimer



bs = 512
in_features = 768
out_features = 3072

input_infos = {"input_0": (bs, in_features)}

output_infos = {"output_0": (bs, out_features)}

parameters = {
    "in_features": in_features,
    "out_features": out_features,
}

gemm_code = importlib.import_module(f"gemm_source.gemm_{bs}_{in_features}_{out_features}")

module_kernel = ModuleKernel(
    "Linear",
    "forward",
    kernel_source=gemm_code.source_code,
    input_infos=input_infos,
    output_infos=output_infos,
    parameters=parameters,
    dependencies=[gemm_code.dependency],
)

# code, _, _, _ = module_kernel.get_code()
# temp_path = BRT_CACHE_PATH / "kernel_template" / "temp.cu"
# temp_path.write_text(code)
kernel_func = module_kernel.get_function()

module_kernel.dump_to_db(rank=11)

#%%

linear = nn.Linear(in_features, out_features, bias=False).cuda()


in_data = torch.ones(bs, in_features).cuda()
manu_out = torch.zeros(bs, out_features).cuda()
tvm_out = torch.zeros(bs, out_features).cuda()

tvm_kernel = make_jit_kernel(linear, in_data)

# weight_t = linear.weight.t().contiguous()

kernel_func(in_data, linear.weight, manu_out)
tvm_kernel(in_data, linear.weight, tvm_out)
torch_result = linear(in_data)

print(torch_result.sum())
print(torch_result)
print(manu_out.sum())
print(manu_out)
print(tvm_out.sum())
print(tvm_out)

timer = CUDATimer(loop=100,repeat=5)

timer.execute(lambda: kernel_func(in_data, linear.weight, manu_out), msg="manu")
timer.execute(lambda: tvm_kernel(in_data, linear.weight, tvm_out), msg="tvm")
timer.execute(lambda: linear(in_data), msg="torch")

# %%
