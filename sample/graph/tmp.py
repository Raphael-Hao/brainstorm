import torch
import torch.nn as nn
from brt.runtime import log
from brt.app.rand import RandScatter
from brt.router import GatherRouter


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_scatter = RandScatter(path_num=2)
        self.expert1 = nn.Identity()
        self.expert2 = nn.Identity()
        self.gather_router = GatherRouter()
        self.iteration = 1
        self.ret = 1

    def forward(self, x):
        route_results = self.rand_scatter(x)
        x_0 = self.expert1(route_results[0])
        x_1 = self.expert2(route_results[1])
        x = self.gather_router([x_0, x_1])
        return x

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = MoE()

    def forward(self, x):
        x = self.moe(x)
        return x


moe_model = SimpleModel()

indata = torch.arange(0, 40, dtype=torch.float32).view(4, 10)
outdata = moe_model(indata)
print(outdata)
from brt.trace.graph import GraphTracer
from torch.fx.graph_module import GraphModule
from brt.runtime import BRT_CACHE_PATH
tracer = GraphTracer()
graph = tracer.trace(moe_model)
# name = moe_model.__class__.__name__ if isinstance(moe_model, torch.nn.Module) else moe_model.__name__
# graph_module= GraphModule(tracer.root, graph, name)

# from torch.fx.passes.graph_drawer import FxGraphDrawer

# graph_drawer = FxGraphDrawer(graph_module, "brt_model")

import pdb;pdb.set_trace()
for node in graph.nodes:
    print(node)

# with open("a.svg", "wb") as f:
#     f.write(graph_drawer.get_dot_graph().create_svg())