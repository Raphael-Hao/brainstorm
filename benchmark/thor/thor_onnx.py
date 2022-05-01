#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /thor_test.py
# \brief:
# Author: raphael hao
#%%
import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

from thor_config import ThorConfig
from thor_model import ThorEncoder, ThorModel


def export_thor_model(model_filename):
    thor_config = ThorConfig()
    thor_config.token_num = 64
    thor_config.runtime = False
    thor_config.hidden_size = 512
    thor_config.intermediate_size = 1024
    thor_config.num_attention_heads = 8
    thor_config.num_hidden_layers = 1
    thor_config.expert_num = 64
    thor_config.expert_type = "sparse_fusion"
    model_filename = (
        thor_config.expert_type
        + "_"
        + str(thor_config.expert_num)
        + "_"
        + model_filename
    )
    thor_encoder = ThorEncoder(thor_config)
    print(thor_encoder)
    thor_encoder.cpu()
    thor_encoder.eval()
    dummy_input = torch.rand(1, 64, 512, device="cpu", dtype=torch.float32)
    dummy_output = thor_encoder(dummy_input)
    torch.onnx.export(thor_encoder, dummy_input, model_filename, opset_version=11)
    print(dummy_output[0].shape)


# def export_bert_model(model_filename):
#     bert_config = BertConfig()
#     bert_config.num_hidden_layers = 1
#     bert_encoder = BertEncoder(bert_config)
#     print(bert_encoder)
#     bert_encoder.cuda()
#     bert_encoder.eval()
#     dummy_input = torch.rand(1, 64, 768, device="cuda", dtype=torch.float32)
#     dummy_output = bert_encoder(dummy_input)
#     torch.onnx.export(bert_encoder, dummy_input, model_filename)
#     print(dummy_output[0].shape)


if __name__ == "__main__":
    export_thor_model("thor_model.onnx")
    # export_bert_model("bert_model.onnx")
# %%
