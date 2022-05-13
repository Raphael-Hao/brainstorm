# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.onnx import register_custom_op_symbolic, symbolic_helper


def router_op(g: torch._C.Graph,n: torch._C.Node, *args, **kwargs ):
    