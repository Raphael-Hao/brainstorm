# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import time

import torch
from brt.common import BRT_KERNEL_TEMPLATE_PATH, log
from brt.jit import HeteroFuseFunction
from brt.jit.compiler import CUDACompiler

log.set_level("jit", "DEBUG")

kernel_name = "sample"

kernel_template_filename = str(BRT_KERNEL_TEMPLATE_PATH / (kernel_name + ".cu"))

kernel_template_source = open(kernel_template_filename, "r").read()

fuser = HeteroFuseFunction([kernel_template_source, kernel_template_source])

code = fuser.get_code()

processed_template_fname = str(
    BRT_KERNEL_TEMPLATE_PATH / ("processed_" + kernel_name + ".cu")
)
with open(processed_template_fname, "w") as f:
    f.write(code)

fused_matmul = CUDACompiler.generate_kernel(None, code)

data_0 = torch.ones((8, 64, 64), device="cuda")
weight_0 = torch.ones((8, 64, 64), device="cuda")
outdata_0 = torch.ones((8, 64, 64), device="cuda")
data_1 = torch.ones((8, 64, 64), device="cuda")
weight_1 = torch.ones((8, 64, 64), device="cuda")
outdata_1 = torch.ones((8, 64, 64), device="cuda")
torch.cuda.synchronize()
stream = torch.cuda.default_stream()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record(stream)
active_blocks = [1, 1]
fused_matmul(
    data_0,
    weight_0,
    outdata_0,
    data_1,
    weight_1,
    outdata_1,
    active_blocks=active_blocks,
)
end_event.record(stream)
stream.synchronize()
print("first time: {:.3f}".format(start_event.elapsed_time(end_event)))

start_event.record(stream)
for i in range(100):
    fused_matmul(
        data_0,
        weight_0,
        outdata_0,
        data_1,
        weight_1,
        outdata_1,
        active_blocks=active_blocks,
    )
end_event.record(stream)
stream.synchronize()
print(outdata_0)
print(outdata_1)
print("forward time: {:.3f}".format(start_event.elapsed_time(end_event) / 100))
