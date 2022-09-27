# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


def print_modules(model: nn.Module):
    for name, module in model.named_modules():
        print(name)
