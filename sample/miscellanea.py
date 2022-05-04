import torch
from brt.common import BRT_KERNEL_TEMPLATE_PATH, log
from brt.jit.block_fuse import BlockFuser
from brt.jit.compiler import CUDACompiler

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

data = torch.ones((8, 64, 64), device="cuda")
weight = torch.ones((8, 64, 64), device="cuda")
outdata = torch.ones((8, 64, 64), device="cuda")
fused_matmul(data, weight, outdata, data, weight, outdata)
