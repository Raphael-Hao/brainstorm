# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
import torch
from torch.fx import symbolic_trace


def script_model(model):
    script_thor_model = torch.jit.script(model)
    print(script_thor_model.graph)
    # for node in script_thor_model.inlined_graph.nodes():
    #     print(node)


def symbolic_trace_model(model):
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)
    print(symbolic_traced.graph)


#%%
import torch.nn as nn
from brt.router import BranchRouter, Router


class DynamicRouting(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = BranchRouter()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.aggregator = Router()

    def forward(self, x):
        x, y = self.router(x) # return torch.zeros with the same shape
        x = self.linear1(x)
        y = self.linear2(y)
        x = self.aggregator(x, y)
        return x
    
    def forward(self, x):
        x, y = self.router(x) # return torch.zeros with the same shape
        x = self.linear1(x)
        y = self.linear2(y)
        x = self.aggregator(x, y)
        return x
    

dynamic_routing = DynamicRouting()
script_model(dynamic_routing)

#%%
from thor_config import ThorConfig
from thor_model import ThorEncoder
from thor_moe import ThorMoE


def create_thor():
    thor_config = ThorConfig()
    thor_config.token_num = 64
    thor_config.runtime = False
    thor_config.hidden_size = 512
    thor_config.intermediate_size = 1024
    thor_config.num_attention_heads = 8
    thor_config.num_hidden_layers = 1
    thor_config.expert_num = 64
    thor_config.expert_type = "router_moe"
    # thor_encoder = ThorEncoder(thor_config)
    thor_encoder = ThorMoE(thor_config)
    return thor_encoder


# %%
thor_model = create_thor()
#%%
script_model(thor_model)
# %%
symbolic_trace_model(thor_model)
# %%
