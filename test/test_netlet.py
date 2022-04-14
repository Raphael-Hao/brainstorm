#%%
import brt
import brt.nn as nn
import torch

#%%
@brt.netlet
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x, y: int):
        for _ in range(y):
            x = self.linear1(x)
            x = self.linear2(x)
        return x


#%%
simple_net = SimpleNet()

x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = simple_net(x, 10)
print(x)

simple_net.brt(True)
script_simple_net = torch.jit.script(simple_net)
simple_net_inlined_graph = script_simple_net.inlined_graph
print(simple_net_inlined_graph)

# %%
