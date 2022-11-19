# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

from brt.passes.base import PassBase, register_pass
from torch.fx.graph import Graph
import torch.nn as nn
from ast import arg
from hashlib import new
from turtle import pd
from typing import Union
import operator
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
                fuse_target=''
                fuse_graphnode=[]
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
                            fuse_graphnode.append([point,bn,activate])
                            fuse_target+=point.target
                            queue[i]=activate
                ## TODO deal with the bug of get item and change the implementation of the logic of add new module
                if len(fuse_sequece)>0:
                    fuselayer=brt.passes.fuse_util.FusedLayer(fuse_sequece,fuse_inputs_shape,fuse_outputs_shape)
                    construct_new_args = []
                    for i,seq in enumerate(fuse_sequece):
                        construct_new_args.append(fuse_graphnode[i][0].args[0])
                    pre_node=outList[-1][-1]
                    with self.graph_mod.graph.inserting_after(pre_node):
                        new_args=(construct_new_args,)
                        self.graph_mod.add_module(fuse_target.replace(".", "_"), fuselayer)
                        self.sub_modules[fuse_target.replace(".", "_")]=fuselayer
                        new_node_insert=self.graph_mod.graph.create_node('call_module', fuse_target.replace(".", "_"), args=new_args, kwargs={})
                    res.append(new_node_insert)       
                    
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
                # if len(fuse_sequece)>0:
                #     for i,seq in enumerate(fuse_sequece):
                #         conv=fuse_graphnode[i][0]
                #         bn= fuse_graphnode[i][1]
                #         activate= fuse_graphnode[i][2]
                #         with self.graph_mod.graph.inserting_after(new_node_insert):
                #             new_gititem=self.graph_mod.graph.call_function(operator.getitem, args=(new_node_insert,i))
                #         activate.replace_all_uses_with(new_gititem)
                #         self.graph_mod.graph.erase_node(activate)
                #         self.graph_mod.graph.erase_node(bn)
                #         self.graph_mod.graph.erase_node(conv)
                    
                outList.append(res)
                queue=nextQueue	
                print("queue",queue)
            return outList

            
            
            
        
        for node in self.graph_mod.graph.nodes:
            out_list=BFS(node)
            break
        for i in range(len(out_list)):
            print("level",i)
            
            fuse_sequece = []
            fuse_inputs_shape=[]
            fuse_outputs_shape=[]
            fuse_target=''
            fuse_graphnode=[]
            queue=out_list[i]
            for j in range(len(queue)):
                point=queue[j]		
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
                        bn_module = self.sub_modules[bn.target]
                        activate_module = self.sub_modules[activate.target]
                        ## TODO construct and deal with the scatter node
                        sequence = nn.Sequential(node_m, bn_module, activate_module)
                        fuse_inputs_shape.append(list(node_m.input_shape))
                        fuse_outputs_shape.append(list(node_m.output_shape))
                        fuse_sequece.append(sequence)
                        fuse_graphnode.append([point,bn,activate])
                        fuse_target+=point.target    
            if len(fuse_sequece)>0:
                for j,seq in enumerate(fuse_sequece):
                    conv=fuse_graphnode[j][0]
                    bn= fuse_graphnode[j][1]
                    activate= fuse_graphnode[j][2]
                    node_insert=out_list[i][-1]
                    with self.graph_mod.graph.inserting_after(node_insert):
                        new_gititem=self.graph_mod.graph.call_function(operator.getitem, args=(node_insert,j))
                    activate.replace_all_uses_with(new_gititem)
                    self.graph_mod.graph.erase_node(activate)
                    self.graph_mod.graph.erase_node(bn)
                    self.graph_mod.graph.erase_node(conv)
            
        self.graph_mod.recompile()
        # from torch.fx.passes.graph_drawer import FxGraphDrawer
        
        # graph_drawer = FxGraphDrawer(self.graph_mod, "raw_pass_graph")
        # with open("hfuse.svg", "wb") as f:
        #     f.write(graph_drawer.get_dot_graph().create_svg())

            
        
    
    
    def finalize(self) -> GraphModule:
        
        ## TODO add topo sort
        def topological():
            in_degrees = dict((u, 0) for u in self.graph_mod.graph.nodes)
            num = len(in_degrees)
            for u in self.graph_mod.graph.nodes:
                for v in u.users.copy().keys():
                    in_degrees[v] += 1
            Q = [u for u in in_degrees if in_degrees[u] == 0]
            Seq = []
            import torch.fx as fx

            while Q:
                u = Q.pop(0)
                Seq.append(u)
                for v in u.users.copy().keys():
                    in_degrees[v] -= 1
                    if in_degrees[v] == 0:
                        Q.append(v)
            if not len(Seq) == num:
                raise Exception("failed to finish topological search")
            return Seq

        def topo_prepend():
            topological_seq = topological()
            i = 0
            for front_node in self.graph_mod.graph.nodes:
                new_node = topological_seq[i]
                if new_node == front_node:
                    i += 1
                    continue
                while not new_node == front_node:
                    front_node.prepend(new_node)
                    i += 1
                    new_node = topological_seq[i]
            return

        topo_prepend()
        self.graph_mod.graph.lint()
        return super().finalize()