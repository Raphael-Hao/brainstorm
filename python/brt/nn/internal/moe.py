# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import brt.nn as nn
from brt.jit.block_fuse import BlockFuser
from brt.primitive import netlet


@netlet
class FusionExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, inputs):
        pass
