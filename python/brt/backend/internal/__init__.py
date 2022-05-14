# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
from brt.common import BRT_PKG_PATH

torch.ops.load_library(BRT_PKG_PATH / "../../build/libbrt_torchscript.so")
