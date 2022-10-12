# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from ast import arg
from hashlib import new
from typing import Union

import torch
from torch.fx import GraphModule
import numpy as np
from brt.passes.base import PassBase, register_pass
from brt.router import ScatterRouter
from brt.router.fabric import make_fabric
from brt.router.protocol import make_protocol


@register_pass("op_transform")
class TransformPass(PassBase):
    def __init__(
        self,
        m: Union[torch.nn.Module, GraphModule],
        dead_load=0.0,
        runtime_load: int = 0,
    ):
        super().__init__(m)
        self.dead_load = dead_load
        self.runtime_load = runtime_load
        
    
    
    def run_on_graph(self):
        i=0
        def bfs(node,fa):
            node_visited={}
            for node_1 in self.graph_mod.graph.nodes:
                node_visited.update({node_1:[]})
            queue = []
            queue.append(node)
            seen = set()
            while len(queue) > 0:
                vetex = queue.pop(0)
                if self.is_placeholder_node(vetex):
                    continue
                nodes = vetex.args[0]
                if self.is_method_node(vetex) and not isinstance(nodes,list):
                    nodes=[]
                    for arg in vetex.args:
                        if isinstance(arg,torch.fx.node.Node):
                            nodes.append(arg)
                elif self.is_function_node(vetex) and not isinstance(nodes,list):
                    nodes=[]
                    for arg in vetex.args:
                        if isinstance(arg,torch.fx.node.Node):
                            nodes.append(arg)
                elif not isinstance(nodes,list):
                    nodes=[nodes]
                for w in nodes:
                    if w not in seen:
                        queue.append(w)
                        seen.add(w)
                    pre=node_visited[w].copy()
                    pre.append(vetex)
                    node_visited.update({w:pre})
            edge_set=set()
            for set_node in seen:
                if len(set_node.users)>len(node_visited[set_node]):
                    if self.is_scatter_node(set_node):
                        continue
                    edge_set.add(set_node)   
            return seen,edge_set,node_visited
        
        for node in self.graph_mod.graph.nodes:
            if self.is_scatter_node(node):
                node_m=self.sub_modules[node.target]
                ## bfs to find the node that have data dependency with node (find the score depandency)
                set_data_dependency,edge_node,node_visited_map=bfs(node.args[0][-1],node)
                edge_users=[]
                edge_nodes_in_seq=[]
                for edge_n in edge_node.copy():
                    if not edge_n in node.args[0]:
                        edge_nodes_in_seq.append(edge_n)
                for edge_n in edge_node.copy():
                    if edge_n in node.args[0]:
                        edge_nodes_in_seq.append(edge_n)
                for edge_n in edge_nodes_in_seq:
                    edge_users.append([])
                    visit_list=node_visited_map[edge_n]
                    for users in edge_n.users:
                        if not users in visit_list:
                            edge_users[-1].append(users)
                edge_node_in_seq_copy=edge_nodes_in_seq.copy()
                i=0
                get_item_node=[]
                for key in node.users.copy().keys():
                    for new_key in key.users.copy().keys():
                        get_item_node.append(new_key)
                import pdb;pdb.set_trace()
                for key in node.users.copy().keys():
                    for new_key in key.users.copy().keys():
                        if i==len(get_item_node)-1:
                            with self.graph_mod.graph.inserting_after(key):
                                new_node=self.graph_mod.graph.call_function(key.target,key.args,key.kwargs)
                                key.replace_all_uses_with(new_node)
                            self.graph_mod.graph.erase_node(key)
                            with self.graph_mod.graph.inserting_after(new_key):
                                new_node=self.graph_mod.graph.call_function(new_key.target,new_key.args,new_key.kwargs)
                                new_key.replace_all_uses_with(new_node)
                            self.graph_mod.graph.erase_node(new_key)
                            import pdb;pdb.set_trace()
                            break 
                        list_front_node=edge_users.pop(0)
                        edge_node_in_seq=edge_nodes_in_seq.pop(0)
                        for new_key1 in new_key.users.copy().keys():
                            if edge_node_in_seq in node.args[0]:
                                index_edge_node:int=node.args[0].index(edge_node_in_seq)
                                if not isinstance(new_key1.args[0],list):
                                    new_args=([get_item_node[index_edge_node]],)
                                else:
                                    construct_new_args=[]
                                    for tmp_node in new_key1.args[0]:
                                        if not tmp_node == new_key:
                                            construct_new_args.append(tmp_node)
                                    construct_new_args.append(get_item_node[index_edge_node])
                                    new_args=(construct_new_args,)
                                new_key1.args=new_args
                                continue
                            if not isinstance(new_key1.args[0],list):
                                new_args=([node.args[0][i]],)
                            else:
                                construct_new_args=[]
                                for tmp_node in new_key1.args[0]:
                                    if not tmp_node == new_key:
                                        construct_new_args.append(tmp_node)
                                construct_new_args.append(node.args[0][i])
                                new_args=(construct_new_args,)
                            new_key1.args=new_args
                        if_continue=False
                        for front_node in list_front_node:
                            if self.is_scatter_node(front_node):
                                if_continue=True
                                break
                        if if_continue:
                            continue
                        for front_node in list_front_node:
                            if not isinstance(front_node.args[0],list):
                                new_args=([new_key],)
                            else:
                                construct_new_args=[]
                                for tmp_node in front_node.args[0]:
                                    if not tmp_node == edge_node_in_seq:
                                        construct_new_args.append(tmp_node)
                                construct_new_args.append(new_key)
                                new_args=(construct_new_args,)
                            front_node.args=new_args
                            if self.is_function_node(front_node):
                                with self.graph_mod.graph.inserting_after(front_node):
                                    new_node=self.graph_mod.graph.call_function(front_node.target,new_args,front_node.kwargs)
                                    front_node.replace_all_uses_with(new_node)
                                self.graph_mod.graph.erase_node(front_node)
                            elif self.is_module_node(front_node):
                                with self.graph_mod.graph.inserting_after(front_node):
                                    new_node=self.graph_mod.graph.call_method(front_node.target,new_args,front_node.kwargs)
                                    front_node.replace_all_uses_with(new_node)
                                self.graph_mod.graph.erase_node(front_node)
                    if i==len(get_item_node)-1:
                        break
                    i=i+1  
                    with self.graph_mod.graph.inserting_after(key):
                        new_node=self.graph_mod.graph.call_function(key.target,key.args,key.kwargs)
                        key.replace_all_uses_with(new_node)
                    self.graph_mod.graph.erase_node(key)
                    with self.graph_mod.graph.inserting_after(new_key):
                        new_node=self.graph_mod.graph.call_function(new_key.target,new_key.args,new_key.kwargs)
                        new_key.replace_all_uses_with(new_node)
                    self.graph_mod.graph.erase_node(new_key)
                    
                    
                    
                
                
                for tmp_node in node.args[0]:
                    if self.is_function_node(tmp_node) and tmp_node.target == torch.zeros_like:
                        edge_node_in_seq_copy.append(tmp_node)
                edge_node_in_seq_copy.append(node.args[0][-1])
                new_args = ([w for w in edge_node_in_seq_copy],node.args[0][-1])
                node.args=new_args
                
                
                with self.graph_mod.graph.inserting_before(node):
                    new_node=self.graph_mod.graph.call_module(node.target,node.args,node.kwargs)
                    node.replace_all_uses_with(new_node)
                self.graph_mod.graph.erase_node(node)
                for node in self.graph_mod.graph.nodes:
                    with  open('/home/yichuanjiaoda/brainstorm_project/brainstorm/python/brt/passes/nodeprocess.txt','a+') as f:
                        f.write(str(i))
                        f.write(str(node))
                        f.write('\n')
                        f.close()
                for node_arg in new_node.args[0]:
                    print('node_arg',node_arg)
                for mm in new_node.users.copy().keys():
                    print('mm {}',mm)
                    for nn in mm.users.copy().keys():
                        print('nn {}',nn)
                        for oo in nn.users.copy().keys():
                            print('oo {}',oo)
                import pdb;pdb.set_trace()
            # self.graph_mod.recompile()
            # self.graph_mod.graph.lint() 
            if self.is_gather_node(node):
                node_m = self.sub_modules[node.target]
        self.graph_mod.recompile()
        self.graph_mod.graph.lint()
        for node in self.graph_mod.graph.nodes:
            with  open('/home/yichuanjiaoda/brainstorm_project/brainstorm/python/brt/passes/node1.txt','a+') as f:
                f.write(str(i))
                f.write(str(node))
                f.write('\n')
                f.close()
        
        with  open('aftertransgraphnode.txt','a+') as f:
                print(self.graph_mod.graph,file=f)
                f.close()
        with  open('aftertransgraphcode.txt','a+') as f:
                print(self.graph_mod.code,file=f)
                f.close()
        import pdb;pdb.set_trace()
        
        return 
    def finalize(self):
        # self.graph_mod.graph.eliminate_dead_code()
        
        self.graph_mod.recompile()
        return self.graph_mod
        return super().finalize()
 