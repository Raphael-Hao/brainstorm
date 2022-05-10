import time

import torch
from brt.common import BRT_KERNEL_TEMPLATE_PATH, log
from brt.jit import HomoFuseFunction
from brt.jit.compiler import CUDACompiler

log.set_level("jit", "DEBUG")

kernel_name = "sample"

fuser = HomoFuseFunction(
    homo_func_name=kernel_name,
    branch_num=2,
    capacities=[1, 2],
    shared_arg_indices=[0, 2],
)

fuser.fuse()
print(fuser.args)

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
shared_inputs = [data_0, outdata_0]
weights=[weight_0, weight_1]
torch.cuda.synchronize()
stream = torch.cuda.default_stream()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record(stream)
active_blocks = [1, 0]
fused_matmul(
    shared_inputs=shared_inputs,
    weights=weights,
    active_blocks=active_blocks,
)
end_event.record(stream)
stream.synchronize()
print("first time: {:.3f}".format(start_event.elapsed_time(end_event)))

start_event.record(stream)
for i in range(100):
    fused_matmul(
    shared_inputs=shared_inputs,
    weights=weights,
    active_blocks=active_blocks,
)
end_event.record(stream)
stream.synchronize()
print("forward time: {:.3f}".format(start_event.elapsed_time(end_event) / 100))
