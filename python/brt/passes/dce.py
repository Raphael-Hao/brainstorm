# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


import torch
import torch.nn as nn
from brt.runtime import log
from brt.app.rand import RandScatter
from brt.router import GatherRouter
from brt.trace.graph import GraphTracer
from torch.fx.graph_module import GraphModule
from brt.runtime import BRT_CACHE_PATH
from torch.fx import Graph
## change from nn model to graph module(actually is a nn model too)
def graph_dce(_module: torch.nn.Module) -> GraphModule:
    
    ##change nn into graph module
    tracer = GraphTracer()
    graph = tracer.trace(_module)
    name = _module.__class__.__name__ if isinstance(_module, torch.nn.Module) else _module.__name__
    graph_module= GraphModule(tracer.root, graph, name)
    
    ## print the graph before dce
    print(graph_module.code)
    from torch.fx.passes.graph_drawer import FxGraphDrawer
    graph_drawer = FxGraphDrawer(graph_module, "brt_model")
    with open("origin.svg", "wb") as f:
        f.write(graph_drawer.get_dot_graph().create_svg())
    
    
    customed_dead_code_elimination(graph)
    
    ## graph module dce built in
    graph.eliminate_dead_code()
    
    ## build new graph module
    new_graph_module = GraphModule(tracer.root, graph, name)
    
    
    ## print the graph after dce
    print(new_graph_module.code)
    from torch.fx.passes.graph_drawer import FxGraphDrawer
    graph_drawer = FxGraphDrawer(new_graph_module, "new_brt_model")
    with open("after.svg", "wb") as f:
        f.write(graph_drawer.get_dot_graph().create_svg())
    return new_graph_module

def customed_dead_code_elimination(graph: Graph) :
    """
    Customized dead code elimination.
    """
    ## todo
    
    for node in graph.nodes:
        if node.target == "moe.gather_router":
            print("delete")
            new_args = ([node.args[0][1]],)
            node.args = new_args
            print(node.args)
            
            





# class MoE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.rand_scatter = RandScatter(path_num=2)
#         self.expert1 = nn.Identity()
#         self.expert2 = nn.Identity()
#         self.gather_router = GatherRouter()
#         self.iteration = 1
#         self.ret = 1

#     def forward(self, x):
#         route_results = self.rand_scatter(x)
#         x_0 = self.expert1(route_results[0])
#         x_1 = self.expert2(route_results[1])
#         x = self.gather_router([x_0, x_1])
#         return x

# class SimpleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.moe = MoE()

#     def forward(self, x):
#         x = self.moe(x)
#         return x
# moe_model = SimpleModel()

# indata = torch.arange(0, 40, dtype=torch.float32).view(4, 10)
# outdata = moe_model(indata)
# print(outdata)
# new_model_= graph_dce(moe_model)