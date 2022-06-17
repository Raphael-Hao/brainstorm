import brt.nn as nn
import torch
from brt.common import log
from brt.routers.app import RandGatherRouter, RandScatterRouter

log.set_level("frontend", "INFO")
log.set_level("backend", "INFO")
log.set_level("ir", "INFO")


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.scatter_router = RandScatterRouter(dst_num=2)
        self.expert1 = nn.Linear(10, 10)
        self.expert2 = nn.Linear(10, 10)
        self.gather_router = RandGatherRouter(dst_num=2)

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

from brt.backend.pytorch import model_to_script
from brt.frontend import build_graph
from brt.primitive.helper import symbolize

symbolized_moe = symbolize(moe_model.eval())
script_moe_model = torch.jit.script(symbolize(moe_model))
sm_graph = script_moe_model.graph
ir_moe_model = build_graph(moe_model)
model_script = model_to_script(ir_moe_model)
print(model_script)
