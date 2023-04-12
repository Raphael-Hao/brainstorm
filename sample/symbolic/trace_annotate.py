# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import torch.nn as nn
from brt import Annotator, symbolic_trace


class AnnotateModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.annotator = Annotator([0])
    def forward(self, x):
        return self.annotator(x)

annotate_module = AnnotateModule()
traced_m = symbolic_trace(annotate_module)
print(traced_m.code)

# %%
