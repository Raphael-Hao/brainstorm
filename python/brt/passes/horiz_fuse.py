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

@register_pass("horiz_fuse")
class HorizFusePass(PassBase):
    jit_need_modules = [nn.Conv2d, nn.Linear]
    native_modules = [nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax]

    def run_on_graph(self):
        ## TODO
        self.graph_mod.graph.eliminate_dead_code()
        
        def bfs(node, fa):
            node_visited = {}
            for node_1 in self.graph_mod.graph.nodes:
                node_visited.update({node_1: []})
            queue = []
            queue.append(node)
            seen = set()
            while len(queue) > 0:
                vetex = queue.pop(0)
                if self.is_placeholder_node(vetex):
                    continue
                nodes = vetex.args[0]
                if self.is_method_node(vetex) and not isinstance(nodes, list):
                    nodes = []
                    for arg in vetex.args:
                        if isinstance(arg, torch.fx.node.Node):
                            nodes.append(arg)
                elif self.is_function_node(vetex) and not isinstance(nodes, list):
                    nodes = []
                    for arg in vetex.args:
                        if isinstance(arg, torch.fx.node.Node):
                            nodes.append(arg)
                elif not isinstance(nodes, list):
                    nodes = [nodes]
                for w in nodes:
                    if w not in seen:
                        queue.append(w)
                        seen.add(w)
                    pre = node_visited[w].copy()
                    pre.append(vetex)
                    node_visited.update({w: pre})
            edge_set = set()
            for set_node in seen:
                if len(set_node.users) > len(node_visited[set_node]):
                    if self.is_scatter_node(set_node):
                        continue
                    edge_set.add(set_node)
            return seen, edge_set, node_visited
        
        
        def BFS(s):
            queue=[s]
            outList=[]
            seen = set()
            node_visited = {}
            for node_1 in self.graph_mod.graph.nodes:
                node_visited.update({node_1: 0})
            while queue:
                ## begin a new level
                res=[]
                nextQueue=[]
                fuse_sequece = []
                fuse_inputs_shape=[]
                fuse_outputs_shape=[]
                for i in range(len(queue)):
                    point=queue[i]		
                    res.append(point)
                    node_visited.update({point: 1})
                    if point.op == 'call_module':
                        node_m = self.sub_modules[point.target]
                        if isinstance(node_m, nn.Conv2d) and node_m.bias is  None:
                            if len(point.users)>1:
                                print("conv2d has users")
                                continue
                            bn,=point.users
                            if len(bn.users)>1:
                                print("bn has users")
                                continue
                            activate,=bn.users
                            node_visited.update({activate: 1})
                            node_visited.update({bn: 1})
                            
                            bn_module = self.sub_modules[bn.target]
                            activate_module = self.sub_modules[activate.target]
                            ## TODO construct and deal with the scatter node
                            sequence = nn.Sequential(node_m, bn_module, activate_module)
                            fuse_inputs_shape.append(list(node_m.input_shape))
                            fuse_outputs_shape.append(list(node_m.output_shape))
                            fuse_sequece.append(sequence)
                            queue[i]=activate
                if len(fuse_sequece)>0:
                    fuselayer=brt.passes.fuse_util.FusedLayer(fuse_sequece,fuse_inputs_shape,fuse_outputs_shape)
                for i in range(len(queue)):
                    point=queue[i]		
                    for son in point.users:
                        if son.target==torch.cat or self.is_scatter_node(son) or self.is_gather_node(son)  or son.target=='view':
                            visitall_cat_arg=True
                            if son.target=='view':
                                iterargs=son.args[:2]
                            else:
                                iterargs=son.args[0]
                            for cat_fa in iterargs:
                                if node_visited[cat_fa]==0:
                                    visitall_cat_arg=False
                            
                            if visitall_cat_arg and node_visited[son]==0:
                                node_visited.update({son: 1})
                                if self.is_scatter_node(son):
                                    assert(len(queue)==1)
                                    for git in son.users:
                                        for git2 in git.users:
                                            nextQueue.append(git2)
                                    break
                                for son_son in son.users:
                                    
                                    # if self.is_scatter_node(son_son):
                                    #     import pdb;pdb.set_trace()
                                    #     continue
                                    if son_son.target==torch.cat or self.is_scatter_node(son_son):
                                        continue
                                        
                                    nextQueue.append(son_son)
                            
                            continue
                        
                        
                        nextQueue.append(son)
                outList.append(res)
                queue=nextQueue	
                print("queue",queue)
            import pdb;pdb.set_trace()
            return outList

            
            
            queue = []
            queue.append(s)
            seen = set()
            seen.add(s)
            level_fuse=[]
            import pdb;pdb.set_trace()
            while len(queue) > 0:
                vetex = queue.pop(0)
                nodes = vetex.users
                vetex_next_level=[]
                for w in nodes:
                    if w not in seen:
                        if w.op == 'call_module':
                            node_m = self.sub_modules[w.target]
                            if isinstance(node_m, nn.Conv2d) and node_m.bias is  None:
                                if len(w.users)>1:
                                    print("conv2d has users")
                                    continue
                                bn,=w.users
                                if len(bn.users)>1:
                                    print("bn has users")
                                    continue
                                activate,=bn.users
                                bn_module = self.sub_modules[bn.target]
                                activate_module = self.sub_modules[activate.target]
                                
                                sequence = nn.Sequential(node_m, bn_module, activate_module)
                                
                                continue 
                            else:
                                queue.append(w)
                                seen.add(w)
                        else:
                            queue.append(w)
                            seen.add(w)
                if len(vetex_next_level)>0:
                    level_fuse.append(vetex_next_level)
                else:
                    continue
                print(vetex)
        
        for node in self.graph_mod.graph.nodes:
            BFS(node)
            break
        
        for node in self.graph_mod.graph.nodes:
            if node.op == 'call_module':
                node_m = self.sub_modules[node.target]
                if isinstance(node_m, nn.Conv2d):
                    # print("conv2d")
                    # print(node)
                    if len(node.users)>1:
                        print("conv2d has users")
                        continue
                    bn,=node.users
                    if len(bn.users)>1:
                        print("bn has users")
                        continue
                    if node_m.bias is not None:
                        continue
                    activate,=bn.users
                    bn_module = self.sub_modules[bn.target]
                    activate_module = self.sub_modules[activate.target]
                    
                    sequence = nn.Sequential(node_m, bn_module, activate_module)
                    ## TODO
                    new_module = brt.passes.fuse_util.TunedKernel(sequence,list(node_m.input_shape),list(node_m.output_shape))
                    
                    # sample_input=torch.randn(node_m.input_shape).cuda()
                    # output=new_module(sample_input)
                    
                    
                    self.sub_modules[node.target]=new_module
                    activate.replace_all_uses_with(node)
                    self.graph_mod.graph.erase_node(activate)
                    self.graph_mod.graph.erase_node(bn)
                    with self.graph_mod.graph.inserting_before(node):
                        self.graph_mod.add_module(node.target.replace(".", "_"), new_module)
                        new_node_insert=self.graph_mod.graph.create_node('call_module', node.target.replace(".", "_"), args=node.args, kwargs=node.kwargs)
                        node.replace_all_uses_with(new_node_insert)
                    
                    self.graph_mod.graph.erase_node(node)
                    
                    self.graph_mod.graph.lint()
                    
                # elif isinstance(node_m, nn.MaxPool2d):
                    # print("maxpool2d")
                    # if len(node.users)>1:
                    #     print("maxpool2d has users")
                    #     continue
                    # activate,=node.users
                    # activate_module = self.sub_modules[activate.target]
                    # sequence = nn.Sequential(node_m, activate_module)
                    # new_module = brt.passes.fuse_util.TunedKernel(sequence,list(node_m.input_shape),list(node_m.output_shape))
                    # self.sub_modules[node.target]=new_module
                    # activate.replace_all_uses_with(node)
                    # self.graph_mod.graph.erase_node(activate)
                    # self.graph_mod.graph.lint()
                    
                
                    
                
            
            
            
            self.graph_mod.recompile()
        
            
        
    
    
    def finalize(self) -> GraphModule:
        self.graph_mod.graph.lint()
        return super().finalize()