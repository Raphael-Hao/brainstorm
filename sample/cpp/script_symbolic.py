#%%
import torch
from backup.symbolic import (
    symbolic_joint_gather_route,
    symbolic_joint_scatter_route,
)


def f(x):
    y = symbolic_joint_scatter_route([x], 1)
    x = symbolic_joint_gather_route(y, 1)
    return x

script_f = torch.jit.script(f)
print(script_f.graph)
# %%
