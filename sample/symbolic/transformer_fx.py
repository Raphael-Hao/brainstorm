#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /transformer.py
# \brief:
# Author: raphael hao
#%%
from transformers.utils import fx as t_fx

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert import BertLayer, BertModel

config = BertConfig(num_hidden_layers=1)

bet_layer = BertModel(config)
graph_bert_layer = t_fx.symbolic_trace(bet_layer)
print(graph_bert_layer.graph)
# %%
