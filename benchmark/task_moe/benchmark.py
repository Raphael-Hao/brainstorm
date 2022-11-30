# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch.distributed as dist
import torch
from transformers import BertGenerationTokenizer
from modeling_bert_generation import BertGenerationDecoder, BertGenerationConfig

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

tokenizer = BertGenerationTokenizer.from_pretrained(
    "google/bert_for_seq_generation_L-24_bbc_encoder"
)
config = BertGenerationConfig()
config.is_decoder = True
config.task_moe = True
config.num_tasks = 16
model = BertGenerationDecoder(config=config).cuda(device)
inputs = tokenizer(
    "Hello, my dog is cute,Hello, my dog is cute,Hello, my dog is cute",
    return_token_type_ids=False,
    return_tensors="pt",
)
#%%

input_ids = inputs["input_ids"].repeat(3, 1).cuda(device)
task_ids = torch.randint(0, 16, (3,)).cuda(device)

#%%
outputs = model(input_ids, task_ids=task_ids)
prediction_logits = outputs.logits
print(prediction_logits.shape)
# %%
print(inputs["input_ids"].dtype)
# %%
