# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

# %%
import brt
import brt.nn as nn

# %%
linear1 = nn.Linear(10, 10)
a = linear1(None)
print(a)
# %%
import torch

b = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = linear1(b)
print(b)
# %%

linear1.brt_script(False)
# %%
a = linear1(None)
print(a)

# %%
linear1.brt_script(True)

script_linear1 = torch.jit.script(linear1)
# %%
print(script_linear1.inlined_graph)
# %%

@brt.netlet
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

simple_net = SimpleNet()


# %%
x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = simple_net(x)
print(x)
# %%
simple_net.brt_script(True)
script_simple_net = torch.jit.script(simple_net)
print(script_simple_net.inlined_graph)
# %%
