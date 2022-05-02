# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
#%%
import torch
from brt.common import BRT_KERNEL_TEMPLATE_PATH
from brt.jit import JitCompiler

kernel_name = "sparse_fusion_2_thor_model"

kernel_template_filename = str(BRT_KERNEL_TEMPLATE_PATH / (kernel_name + ".cu"))

kernel_template_source = open(kernel_template_filename, "r").read()
# print(kernel_template_source)
kernel_func = JitCompiler.generate_kernel(
    keyword_dict=None, template=kernel_template_source
)
data = torch.ones((8, 64, 64), device="cuda")
weight = torch.ones((8, 64, 64), device="cuda")
outdata = torch.ones((8, 64, 64), device="cuda")
kernel_func(data, weight, outdata)
# %%
