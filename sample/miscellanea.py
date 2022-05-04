import torch
from brt.common import BRT_KERNEL_TEMPLATE_PATH, log
from brt.jit.block_fuse import BlockFuser
from brt.jit.compiler import CUDACompiler
from brt.jit.generic import GenericFunction

log.set_level("jit", "DEBUG")

kernel_name = "sample"

kernel_template_filename = str(BRT_KERNEL_TEMPLATE_PATH / (kernel_name + ".cu"))

kernel_template_source = open(kernel_template_filename, "r").read()

block_fuser = BlockFuser([kernel_template_source, kernel_template_source])

code = block_fuser.get_code()

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
fused_matmul(data_0, weight_0, outdata_0, data_1, weight_1, outdata_1)

print(outdata_0)
print(outdata_1)
