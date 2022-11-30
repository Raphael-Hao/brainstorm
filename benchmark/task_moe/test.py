# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

#%%
from transformers import BertGenerationTokenizer
from modeling_bert_generation import BertGenerationDecoder, BertGenerationConfig
tokenizer = BertGenerationTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
config.is_decoder = True
model = BertGenerationDecoder.from_pretrained(
    "google/bert_for_seq_generation_L-24_bbc_encoder", config=config
)
inputs = tokenizer("Hello, my dog is cute,Hello, my dog is cute,Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")
#%%

input_ids = inputs["input_ids"].repeat(3, 1)
print(input_ids.shape)

#%%
print(inputs["input_ids"].shape)
outputs = model(input_ids)
prediction_logits = outputs.logits
print(prediction_logits.shape)
# %%
print(inputs["input_ids"].dtype)
# %%
