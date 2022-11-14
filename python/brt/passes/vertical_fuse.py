# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.passes.base import PassBase, register_pass
from torch.fx.graph import Graph
import torch.nn as nn
from ast import arg
from hashlib import new
from turtle import pd
from typing import Union

import torch
from torch.fx import GraphModule
import numpy as np
from brt.passes.base import PassBase, register_pass
from brt.router import ScatterRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol
import brt.passes.fuse_util

@register_pass("vertical_fuse")
class VerticalFusePass(PassBase):
    jit_need_modules = [nn.Conv2d, nn.Linear]
    native_modules = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax]
    

    def run_on_graph(self):
        ## TODO
        for node in self.graph_mod.graph.nodes:
            if node.op == 'call_module':
                node_m = self.sub_modules[node.target]
                if isinstance(node_m, nn.Conv2d):
                    import pdb; pdb.set_trace()
                    print("conv2d")
                    if len(node.users)>0:
                        print("conv2d has users")
                        continue
                    bn,=node.users
                    if len(bn.users)>0:
                        print("bn has users")
                        continue
                    activate,=bn.users
                    bn_module = self.sub_modules[bn.target]
                    activate_module = self.sub_modules[activate.target]
                    
                    sequence = nn.Sequential(node_m, bn_module, activate_module)
                    ## TODO
                    new_module = brt.passes.fuse_util.TunedKernel(sequence,list(node_m.input_shape),list(node_m.output_shape))
                    
                    
                    self.sub_modules[node.target]=new_module
                    activate.replace_all_uses_with(node)
                    self.graph_mod.graph.erase_node(bn)
                    self.graph_mod.graph.erase_node(activate)
                    
                    
                
            
            
            
            self.graph_mod.recompile()
        
    
    
    
    def finalize(self) -> GraphModule:
        self.graph_mod.graph.lint()
        return super().finalize()