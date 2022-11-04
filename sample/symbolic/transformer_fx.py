#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /transformer.py
# \brief:
# Author: raphael hao
#%%
from transformers.utils import fx as t_fx

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert import BertLayer, BertModel, BertTokenizer
import traceback
import torch
# from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes import graph_drawer
import graphviz
from mono_infer import ShapeProp

config = BertConfig(num_hidden_layers=1)

bert_layer = BertModel(config)
graph_bert_layer = t_fx.symbolic_trace(bert_layer)
# print(graph_bert_layer.graph)
# # %%



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text="Hello, my dog is cute"
encoded_input = tokenizer(text, max_length=100,
                          add_special_tokens=True, truncation=True,
                          padding=True, return_tensors="pt")
print(encoded_input)

positional_inputs = (encoded_input['input_ids'], encoded_input['attention_mask'], encoded_input['token_type_ids'], None, None, None, None, None, None, None, None, None, None)

# ShapeProp(graph_bert_layer).propagate(*positional_inputs)
ShapeProp(graph_bert_layer).propagate(*positional_inputs)

g = graph_drawer.FxGraphDrawer(graph_bert_layer, 'bert_layer')
g.get_main_dot_graph().write_svg("bert_layer.svg")
