# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time

import datasets
import numpy as np
import torch
from nvitop import Device
from switch_transformer import (
    SwitchTransformersModel,
    FusedSwitchTransformersSparseMLP,
    BatchmatmulSwitchTransformersSparseMLP,
)  # v4.25.1
from transformers import AutoConfig, AutoTokenizer
from config import SwitchTransformersConfig
from brt.runtime.benchmark import profile_v2


def get_gpu_info():
    devices = Device.all()  # or `Device.cuda.all()` to use CUDA ordinal instead
    for device in devices[:1]:
        processes = device.processes()
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


def main():
    parser = argparse.ArgumentParser(description="Basic")
    parser.add_argument(
        "--expert", type=int, default=8, choices=[8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument(
        "--mode", type=str, default="throughput", choices=["throughput", "debug", "profile"]
    )
    parser.add_argument(
        "--vendor", type=str, default="torch", choices=["torch", "brt", "batchmatmul"]
    )
    args = parser.parse_args()

    model_name = f"google/switch-base-{args.expert}"
    config: SwitchTransformersConfig = AutoConfig.from_pretrained(
        model_name, max_position_embeddings=args.max_seq_length
    )
    config.capacities = [
        # 2,  # 1,
        # 4,  # 1,
        # 8,  # 1,
        16,  # 1,
        32,  # 1,
        64,  # 1,
        128,  # 1,
        224,  # 1,
        320,  # 1, no need may
        416,  # 1,
        512,
    ]
    config.ranks = [
        [
            # 1,  # 2
            # 2,  # 4
            # 1,  # 8
            1,  # 16
            1,  # 32
            2,  # 64
            9,  # 128
            2,  # 224
            1,  # 320
            2,  # 416
            1,  # 512
        ],
        [
            # 1,  # 2
            # 1,  # 4
            # 1,  # 8
            1,  # 16
            1,  # 32
            1,  # 64
            1,  # 128
            1,  # 224
            5,  # 320
            1,  # 416
            3,  # 512
        ],
    ]
    config.vendor = args.vendor

    model = SwitchTransformersModel.from_pretrained(model_name, config=config).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    for _name, m in model.named_modules():
        if isinstance(m, FusedSwitchTransformersSparseMLP) or isinstance(
            m, BatchmatmulSwitchTransformersSparseMLP
        ):
            m.initialize_fused_expert()

    if args.mode == "debug":
        debug(args, model, tokenizer)
    elif args.mode == "throughput":
        throughput(args, model, tokenizer, 100)
    elif args.mode == "profile":
        profile(args, model, tokenizer, args.vendor)

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


def load_data(args, tokenizer, data_num=100):
    dataset = datasets.load_dataset("glue", "mnli")
    all_data_num = len(dataset["train"])
    seed = 171
    np.random.seed(seed)
    datas = []
    for _idx in range(data_num):
        idx = np.random.randint(0, all_data_num)
        inputs = [
            dataset["train"][ii + idx]["premise"]
            + "</s>"
            + dataset["train"][ii + idx]["hypothesis"]
            for ii in range(args.bsz)
        ]
        inputs = tokenizer(
            inputs,
            padding="max_length",
            max_length=args.max_seq_length,
            truncation=True,
        )
        inputs = {ii: torch.tensor(jj).cuda() for ii, jj in inputs.items()}
        inputs["decoder_input_ids"] = inputs["input_ids"]
        datas.append(inputs)
    return datas


def throughput(args, model, tokenizer, test_data_num):
    loaded_data = load_data(args, tokenizer)
    data_num = len(loaded_data)
    random_idx = np.random.choice(range(data_num), test_data_num)
    model(**loaded_data[0])
    with torch.inference_mode():
        torch.cuda.synchronize()
        st = time.time()
        for ii, idx in enumerate(random_idx):
            model(**loaded_data[idx])
            # if ii == 1000 // 2:
            #     get_gpu_info()
        torch.cuda.synchronize()
        end = time.time()
        print("Forward Implementation", end - st)


def debug(args, model, tokenizer):
    loaded_data = load_data(args, tokenizer, 1)
    torch.cuda.synchronize()
    for in_data in loaded_data:
        out = model(**in_data)
        print(out[0])
        print(out[0].sum())
    torch.cuda.synchronize()


def profile(args, model, tokenizer, vendor):
    loaded_data = load_data(args, tokenizer, 10)
    model(**loaded_data[0])
    torch.cuda.synchronize()
    profile_v2(model, loaded_data, vendor)


if __name__ == "__main__":
    main()