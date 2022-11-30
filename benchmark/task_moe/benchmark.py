# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch.distributed as dist
import torch
import torch.nn as nn
from transformers import BertGenerationTokenizer
from modeling_bert_generation import BertGenerationDecoder, BertGenerationConfig
from brt.runtime.benchmark import BenchmarkArgumentManager


def main():
    arg_manager = BenchmarkArgumentManager()
    parser = arg_manager.get_parser()
    parser.add_argument(
        "--mode", type=str, default="debug", choices=["debug", "throughput"]
    )
    args = parser.parse_args()
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
    config.placement_aware = False
    model = BertGenerationDecoder(config=config).cuda(device)
    inputs = tokenizer(
        "Hello, my dog is cute,Hello",
        return_token_type_ids=False,
        return_tensors="pt",
    )
    #%%

    input_ids = inputs["input_ids"].repeat(16, 1).cuda()

    #%%
    model_ddp = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, bucket_cap_mb=4
    )
    if args.mode == "debug":
        debug(config, model_ddp, input_ids)
    elif args.mode == "throughput":
        throughput(config, model_ddp, input_ids)


def debug(config: BertGenerationConfig, model_ddp: nn.Module, input_ids: torch.Tensor):
    local_rank = dist.get_rank()
    model_ddp.eval()
    with torch.inference_mode():
        all_task_ids = [
            [1, 0, 0, 2, 1, 1, 1, 0, 0, 2, 1, 1, 3, 0, 3, 2],
            [0, 1, 2, 3, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
        ]
        task_ids = torch.tensor(all_task_ids[local_rank], dtype=torch.int64).cuda()
        for i in range(4):
            # print(f"local_rank: {local_rank}, input_ids.shape: {input_ids.shape}")
            # print(f"local_rank: {local_rank}, task_ids: {task_ids}")
            outputs = model_ddp(input_ids, task_ids=task_ids)
            prediction_logits = outputs.logits
            print(f"logits: {prediction_logits.sum()}")
            print(prediction_logits.shape)


def throughput(
    config: BertGenerationConfig, model_ddp: nn.Module, input_ids: torch.Tensor
):
    model_ddp.eval()
    local_rank = dist.get_rank()
    with torch.inference_mode():
        while True:
            task_ids = torch.randint(0, config.num_tasks, (input_ids.size(0),)).cuda()
            print(f"local_rank: {local_rank}, input_ids.shape: {input_ids.shape}")
            print(f"local_rank: {local_rank}, task_ids: {task_ids}")
            outputs = model_ddp(input_ids, task_ids=task_ids)
            prediction_logits = outputs.logits
            print(prediction_logits.shape)


if __name__ == "__main__":
    main()
# %%
