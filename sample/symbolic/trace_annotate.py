# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import torch
import torch.nn as nn
import brt
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
sub_modules = dict(traced_m.named_modules())
for i, node in enumerate(traced_m.graph.nodes):
    if node.op == "call_module":
        if node.target == "annotator":
            node_m = sub_modules[node.target]
            with traced_m.graph.inserting_after(node):
                to_torch_node = traced_m.graph.call_function(brt.to_torch_tensor, (node,))
            with traced_m.graph.inserting_after(to_torch_node):
                to_brt_node = traced_m.graph.call_function(brt.to_grid_tensor, (to_torch_node,))
            node.replace_all_uses_with(to_brt_node,lambda usr: usr != to_torch_node)
traced_m.recompile()
print(traced_m.code)

x = torch.randn(1, 3, 224, 224)
traced_m(x)

# %%
