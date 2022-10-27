# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from brt.router import is_router
import numpy as np


def dump_trace(mod: nn.Module):
    scatter_results = []
    for m_name, m in mod.named_modules():
        if is_router(m) and "scatter" in m._router_type:
            scatter_results.append(np.array(m.ptu_decision_history, dtype=object))
    np.save("scatter_results.npy", scatter_results, allow_pickle=True)
