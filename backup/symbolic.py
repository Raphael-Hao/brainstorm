# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable

import torch
from brt.common import find_lib_path

torch.ops.load_library(find_lib_path("libbrt_torchscript.so")[0])

symbolic_scatter_route = torch.ops.brt.symbolic_scatter_route
symbolic_gather_route = torch.ops.brt.symbolic_gather_route
symbolic_joint_scatter_route = torch.ops.brt.symbolic_joint_scatter_route
symbolic_joint_gather_route = torch.ops.brt.symbolic_joint_gather_route
