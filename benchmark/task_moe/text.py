# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
import torch
tokenizer = BertGenerationTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
config.is_decoder = True
model = BertGenerationDecoder.from_pretrained(
    "google/bert_for_seq_generation_L-24_bbc_encoder", config=config
)
inputs = tokenizer("Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")
print(inputs["input_ids"].shape)
outputs = model(**inputs)
prediction_logits = outputs.logits
print(prediction_logits.shape)
# %%
