#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /text_gen.py
# \brief:
# Author: raphael hao

# %%
import torch
import time

device = torch.device("cuda:0")

# %%
from transformers import BertTokenizer
from modeling_albert import AlbertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("/home/whcui/checkpoints/albert_chinese_tiny")
model = AlbertForMaskedLM.from_pretrained("/home/whcui/checkpoints/albert_chinese_tiny")
model.to(device)

# %%
# from transformers import EncoderDecoderModel, RobertaTokenizer

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#     "roberta-base", "roberta-base", pad_token_id=tokenizer.eos_token_id
# )
# model.to(device)

# %%
prompt = "I am very happy to meet you"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to(device)
start_time = time.time()
for i in range(20):
    outputs = model.generate(input_ids, num_beams=32, do_sample=False, max_length=64)
torch.cuda.synchronize(device=device)
compute_time = time.time() - start_time
print("eos based inference time: {:.4f} s".format(compute_time / 20))
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

start_time = time.time()
for i in range(20):
    outputs = model.generate(
        input_ids, num_beams=32, do_sample=False, max_length=64, no_wait=True
    )
torch.cuda.synchronize(device=device)
compute_time = time.time() - start_time
print("no wait inference time: {:.4f} s".format(compute_time / 20))
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# %%
