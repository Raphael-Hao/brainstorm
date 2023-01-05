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
    SwitchTransformersSparseMLP,
)  # v4.25.1
from transformers import AutoConfig, AutoTokenizer
from config import SwitchTransformersConfig
from brt.runtime.benchmark import profile_v2
from brt.runtime import BRT_CACHE_PATH


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
        "--mode",
        type=str,
        default="throughput",
        choices=["throughput", "debug", "profile", "trace"],
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
        96,  # 1,
        128,  # 1,
        160,  # 1,
        192,  # 1,
        # 224,  # 1,
        # 272,  # 1,
        # 320,  # 1, no need may
        # 368,  # 1,
        # 416,  # 1,
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
            1,  # 96
            1,  # 128
            3,  # 160
            1,  # 192
            # 2,  # 224
            # 1,  # 272
            # 1,  # 320
            # 1,  # 368
            # 2,  # 416
            1,  # 512
        ],
        [
            # 1,  # 2
            # 1,  # 4
            # 1,  # 8
            1,  # 16
            1,  # 32
            1,  # 64
            1,  # 96
            1,  # 128
            1,  # 160
            1,  # 192
            # 1,  # 224
            # 1,  # 272
            # 5,  # 320
            # 1,  # 368
            # 1,  # 416
            3,  # 512
        ],
    ]
    config.vendor = args.vendor
    config.trace = args.mode == "trace"

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
        throughput(args, model, tokenizer, 1000)
    elif args.mode == "profile":
        profile(args, model, tokenizer, args.vendor)
    elif args.mode == "trace":
        trace(args, model, tokenizer)
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


def throughput(args, model, tokenizer, test_data_num, warmup=10):
    loaded_data = load_data(args, tokenizer)
    data_num = len(loaded_data)
    result_path = BRT_CACHE_PATH / "results" / "switch_transformer"
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = result_path / "e2e.csv"
    result = result_file.open("a")
    random_idx = np.random.choice(range(data_num), test_data_num)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    model(**loaded_data[0])
    torch.cuda.synchronize()
    with torch.inference_mode():
        for i in range(warmup):
            model(**loaded_data[i])
        print("warmup done, start benchmarking")
        torch.cuda.synchronize()
        start_event.record()
        for ii, idx in enumerate(random_idx):
            model(**loaded_data[idx])
        end_event.record()
        end_event.synchronize()
    result.write(
        f"{args.vendor},{args.bsz},{args.max_seq_length},{args.expert},{(start_event.elapsed_time(end_event) / test_data_num):.2f}\n"
    )


def debug(args, model, tokenizer):
    loaded_data = load_data(args, tokenizer, 1)
    torch.cuda.synchronize()
    for in_data in loaded_data:
        out = model(**in_data)
        print(out[0])
        print(out[0].sum())
    torch.cuda.synchronize()


def trace(args, model, tokenizer):
    loaded_data = load_data(args, tokenizer)
    data_num = len(loaded_data)
    result_path = BRT_CACHE_PATH / "results" / "switch_transformer"
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = result_path / f"track_{args.expert}.csv"
    with torch.inference_mode():
        for idx in range(data_num):
            model(**loaded_data[idx])

    all_history = []
    for _name, m in model.named_modules():
        if isinstance(m, SwitchTransformersSparseMLP):
            np.set_printoptions(linewidth=300)
            average_load = np.min(m.shape_history, axis=1).astype(int)
            # average_load = np.average(m.shape_history, axis=1).astype(int)
            # average_load = np.max(m.shape_history, axis=1).astype(int)
            all_history.append(average_load)
    np.savetxt(result_file, np.array(all_history), delimiter=",", fmt="%d")

def profile(args, model, tokenizer, vendor):
    loaded_data = load_data(args, tokenizer, 10)
    model(**loaded_data[0])
    torch.cuda.synchronize()
    profile_v2(model, loaded_data, vendor)


if __name__ == "__main__":
    main()
