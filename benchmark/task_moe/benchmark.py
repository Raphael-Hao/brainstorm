# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch.distributed as dist
import torch
from transformers import BertGenerationTokenizer
from modeling_bert_generation import BertGenerationDecoder, BertGenerationConfig

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
print(f"local_rank: {local_rank}, world_size: {world_size}")
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

tokenizer = BertGenerationTokenizer.from_pretrained(
    "google/bert_for_seq_generation_L-24_bbc_encoder"
)
config = BertGenerationConfig()
config.is_decoder = True
config.task_moe = True
config.num_tasks = 4
model = BertGenerationDecoder(config=config).cuda(device)
inputs = tokenizer(
    "Hello, my dog is cute,Hello",
    return_token_type_ids=False,
    return_tensors="pt",
)
#%%

input_ids = inputs["input_ids"].repeat(3, 1).cuda()
task_ids = torch.randint(0, config.num_tasks, (3,)).cuda()
print(f"local_rank: {local_rank}, input_ids.shape: {input_ids.shape}")
print(f"local_rank: {local_rank}, task_ids.shape: {task_ids.shape}")

#%%
model_ddp = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[local_rank], broadcast_buffers=False, bucket_cap_mb=4
)
model_ddp.eval()
print(input_ids.shape)
print(task_ids.shape)
with torch.inference_mode():
    outputs = model_ddp(input_ids, task_ids=task_ids)
outputs = model_ddp(input_ids, task_ids=task_ids)
prediction_logits = outputs.logits
print(prediction_logits.shape)
# %%