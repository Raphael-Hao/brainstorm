# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time

import datasets
import numpy as np
import torch
from nvitop import Device
from switch_transformer import SwitchTransformersModel, SwitchTransformersSparseMLP  # v4.25.1
from transformers import AutoConfig, AutoTokenizer
from brt.runtime import BRT_CACHE_PATH


def get_gpu_info():
    devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
    for device in devices[:1]:
        processes = device.processes()  # type: Dict[int, GpuProcess]
        sorted_pids = sorted(processes.keys())
        print(device)
        print(f"  - Fan speed:       {device.fan_speed()}%")
        print(f"  - Temperature:     {device.temperature()}C")
        print(f"  - GPU utilization: {device.gpu_utilization()}%")
        print(f"  - Total memory:    {device.memory_total_human()}")
        print(f"  - Used memory:     {device.memory_used_human()}")
        print(f"  - Free memory:     {device.memory_free_human()}")
        print(f"  - Processes ({len(processes)}): {sorted_pids}")
        for pid in sorted_pids:
            print(f"    - {processes[pid]}")
        print("-" * 120)


data_dir = BRT_CACHE_PATH / "dataset/glue/"
test_time = 100
seed = 171
max_seq_length = 128
bsz = 8
model_name = "google/switch-base-256"
device = torch.device("cuda:0")
config = AutoConfig.from_pretrained(model_name, max_position_embeddings=max_seq_length)
model = SwitchTransformersModel.from_pretrained(model_name, config=config).cuda()
model.eval()
# Load to different GPU
# model.encoder.embed_tokens.to("cuda:0")
# for ii in range(6):
#     model.encoder.block[ii].to("cuda:0")
# for ii in range(6, 12):
#     model.encoder.block[ii].to("cuda:1")
# model.encoder.final_layer_norm.to("cuda:1")
# # model.decoder.embed_tokens.to("cuda:2")
# for ii in range(6):
#     model.decoder.block[ii].to("cuda:2")
# for ii in range(6, 12):
#     model.decoder.block[ii].to("cuda:3")
# model.decoder.final_layer_norm.to("cuda:3")
parser = argparse.ArgumentParser(description="Basic")
parser.add_argument("--dataset_name", type=str, default="")
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(model_name)
d = datasets.load_dataset("glue", "mnli")
N = len(d["train"])
np.random.seed(seed)
datas = []
for ii in range(100):
    idx = np.random.randint(0, N)
    inputs = [
        d["train"][ii + idx]["premise"] + "</s>" + d["train"][ii + idx]["hypothesis"]
        for ii in range(bsz)
    ]
    inputs = tokenizer(
        inputs, padding="max_length", max_length=max_seq_length, truncation=True
    )
    inputs = {ii: torch.tensor(jj).to(device) for ii, jj in inputs.items()}
    inputs["decoder_input_ids"] = inputs["input_ids"]
    datas.append(inputs)
N = len(datas)
random_idx = np.random.choice(range(N), 1000)
torch.cuda.synchronize()
st = time.time()
for ii, idx in enumerate(random_idx):
    model(**datas[idx])
    if ii == 1000 // 2:
        get_gpu_info()
np.set_printoptions(linewidth=1000)
for name, mod in model.named_modules():
    if isinstance(mod, SwitchTransformersSparseMLP):
        print(f"=================={name}==================")
        print(f"Max Tokens: {mod.shape_history.max(axis=1)}")
        print(f"P90 Tokens: {np.percentile(mod.shape_history, 90, axis=1).astype(int)}")
        print(f"P50 Tokens: {np.percentile(mod.shape_history, 50, axis=1).astype(int)}")
        print(f"P30 Tokens: {np.percentile(mod.shape_history, 10, axis=1).astype(int)}")
        print(f"P10 Tokens: {np.percentile(mod.shape_history, 10, axis=1).astype(int)}")
        print(f"Min Tokens: {mod.shape_history.min(axis=1)}")
torch.cuda.synchronize()
end = time.time()
print("Forward Implementation", end - st)
