import torch
from brt.common import find_lib_path

torch.ops.load_library(find_lib_path("libbrt_torchscript.so")[0])

symbolic_scatter_route = torch.ops.brt.symbolic_scatter_route
symbolic_gather_route = torch.ops.brt.symbolic_gather_route
