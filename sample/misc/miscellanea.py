import torch
import torch.nn as nn
from brt.app import RandScatter
from brt.router import GatherRouter


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scatter_router = RandScatter(rand_path_num=2)
        self.expert1 = nn.Linear(10, 10)
        self.expert2 = nn.Linear(10, 10)
        self.gather_router = GatherRouter()

    def forward(self, x):
        route_results = self.scatter_router(x)
        x_0 = self.expert1(route_results[0])
        x_1 = self.expert2(route_results[1])
        x = self.gather_router([x_0, x_1])
        return x


class MoEModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.moe = MoE()

    def forward(self, x):
        return self.moe(x)


moe_model = MoEModel()
x = torch.arange(0, 30, dtype=torch.float32).view(3, 10)
x = moe_model(x)
print(x)
