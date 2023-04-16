from typing import List, Tuple, Union, Literal, Optional

import torch
from torch import nn

from brt.jit.modules import AtomModule
from brt.jit.codegen.module import ModuleKernel


class Conv2dMulAdd(nn.Module):
    def __init__(
        self,
        conv2d,
        scale=1,
    ) -> None:
        super().__init__()
        self.conv2d = conv2d
        self.scale = scale

    def forward(self, x: torch.Tensor, add: torch.Tensor):
        x = self.conv2d(x)
        if self.scale != 1:
            x = x * self.scale
        x = x + add
        return x

