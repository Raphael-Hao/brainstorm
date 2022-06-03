import brt
import brt.nn as nn
import torch
from brt.common import log

# from brt.frontend import build_graph
from brt.router import TopKScatterRouter

log.set_level("router", "DEBUG")
# log.set_level("primitive", "DEBUG")
# log.set_level("frontend", "DEBUG")
# log.set_level("backend", "DEBUG")
# log.set_level("ir", "DEBUG")


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scatter_router = TopKScatterRouter(route_num=2, gran_dim=10)
        self.expert1 = nn.Linear(10, 10)
        self.expert2 = nn.Linear(10, 10)

    def forward(self, x):
        route_results, reverse_indice, reverse_shape = self.scatter_router(x)
        x_0 = self.expert1(route_results[0])
        x_1 = self.expert2(route_results[1])
        x = x_0 + x_1
        return x


class MoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = MoE()

    def forward(self, x):
        return self.moe(x)


moe_model = MoEModel()

# ir_model = build_graph(moe_model)

x = torch.ones((60, 10))
x = x.cuda()
moe_model.cuda()
z = moe_model(x)
print(z)
