# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
import torch

class Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active_counter = 0

    @torch.jit.ignore
    def forward(self, *inputs):
        return self.route(*inputs)

    def route(self, *inputs):
        raise NotImplementedError

    def record(self):
        raise NotImplementedError

    def register_router(self):
        pass
