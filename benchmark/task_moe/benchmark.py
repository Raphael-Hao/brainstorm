# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist
import torch.nn as nn
from brt.runtime.benchmark import BenchmarkArgumentManager, ResultWriter
from brt.runtime.placement import dump_decision
from modeling_bert_generation import BertGenerationConfig, BertGenerationDecoder
from transformers import BertGenerationTokenizer


def main():
    arg_manager = BenchmarkArgumentManager()
    parser = arg_manager.get_parser()
    parser.add_argument(
        "--mode", type=str, default="debug", choices=["debug", "throughput", "trace"]
    )
    parser.add_argument(
        "--opt", type=str, default="None", choices=["None", "placement", "pytorch"]
    )
    parser.add_argument("--seq", type=int, choices=[256, 512])
    parser.add_argument("--token", type=int, choices=[32, 64])
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
    config.num_tasks = 16
    if args.opt == "pytorch":
        config.pt_native = True
    else:
        config.pt_native = False
        if args.opt == "placement":
            config.placement_aware = True
        else:
            config.placement_aware = False
    model = BertGenerationDecoder(config=config).cuda(device)
    inputs_64 = tokenizer(
        "To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing other tasks we performed experiments on English constituency parsing performed experiments on English To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing other tasks we performed experiments on English constituency parsing performed experiments on English",
        return_token_type_ids=False,
        return_tensors="pt",
    )
    inputs_32 = tokenizer(
        "To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing other tasks we performed experiments on English constituency parsing performed experiments on English",
        return_token_type_ids=False,
        return_tensors="pt",
    )
    # %%
    if args.mode == "debug":
        args.seq = 4
    # input_ids = inputs["input_ids"].repeat(4, 1).cuda()
    if args.token == 32:
        input_ids = inputs_32["input_ids"].repeat(args.seq, 1).cuda()
    elif args.token == 64:
        input_ids = inputs_64["input_ids"].repeat(args.seq, 1).cuda()
    print(f"local_rank: {local_rank}, input_ids.shape: {input_ids.shape}")

    # %%
    model_ddp = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, bucket_cap_mb=4
    )
    if args.mode == "debug":
        debug(config, model_ddp, input_ids)
    elif args.mode == "throughput":
        throughput(config, model_ddp, input_ids)
    elif args.mode == "trace":
        trace(config, model_ddp, model, input_ids)


def debug(config: BertGenerationConfig, model_ddp: nn.Module, input_ids: torch.Tensor):
    local_rank = dist.get_rank()
    model_ddp.eval()
    with torch.inference_mode():
        all_task_ids = [
            [
                1,
                0,
                3,
                2,
            ],
            [
                0,
                3,
                2,
                1,
            ],
        ]
        task_ids = torch.tensor(all_task_ids[local_rank], dtype=torch.int64).cuda()
        for i in range(10):
            # print(f"local_rank: {local_rank}, input_ids.shape: {input_ids.shape}")
            # print(f"local_rank: {local_rank}, task_ids: {task_ids}")
            outputs = model_ddp(input_ids, task_ids=task_ids)
            prediction_logits = outputs.logits
            print(f"logits: {prediction_logits.sum()}")
            print(prediction_logits.shape)


def throughput(
    config: BertGenerationConfig, model_ddp: nn.Module, input_ids: torch.Tensor
):
    bench_iteration = 100
    model_ddp.eval()
    all_task_ids = []
    torch.random.manual_seed(dist.get_rank())
    # num_per_task = input_ids.size(0) // config.num_tasks
    # task_ids = torch.arange(config.num_tasks, dtype=torch.int64).repeat(num_per_task)
    # for _ in range(bench_iteration):
    #     all_task_ids.append(task_ids[torch.randperm(task_ids.size(0))])
    for _ in range(bench_iteration):
        all_task_ids.append(torch.randint(0, config.num_tasks, (input_ids.size(0),)))
    result_fname = "task_moe/throughput.csv"
    result_writer = ResultWriter(result_fname)
    with torch.inference_mode():
        for i in range(20):
            outputs = model_ddp(input_ids, task_ids=all_task_ids[i])
        torch.cuda.synchronize()

        if dist.get_rank() == 0:
            print("warmup done, start benchmark")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start_event.record(torch.cuda.current_stream())
        for i in range(bench_iteration):
            outputs = model_ddp(input_ids, task_ids=all_task_ids[i])
        end_event.record(torch.cuda.current_stream())
        end_event.synchronize()
        benched_throughput = (
            bench_iteration
            * input_ids.size(0)
            / start_event.elapsed_time(end_event)
            * 1000
        )
        if config.pt_native:
            item = "Torch"
        else:
            if config.placement_aware:
                item = "BRT+P"
            else:
                item = "BRT"
        GPUS = dist.get_world_size()
        seq_num = input_ids.size(0)
        token_num = input_ids.size(1)
        result_writer.write(
            f"{item},{GPUS}GPUx{int(16/GPUS)}E,{seq_num}Seqsx{token_num}Tokens,{benched_throughput}"
        )
        print(
            f"local_rank: {dist.get_rank()}, throughput: {benched_throughput} samples/s"
        )


def trace(
    config: BertGenerationConfig,
    model_ddp: nn.Module,
    model: nn.Module,
    input_ids: torch.Tensor,
):
    bench_iteration = 100
    model_ddp.eval()
    all_task_ids = []
    torch.random.manual_seed(dist.get_rank())
    for _ in range(bench_iteration):
        all_task_ids.append(torch.randint(0, config.num_tasks, (input_ids.size(0),)))
    with torch.inference_mode():
        for i in range(bench_iteration):
            outputs = model_ddp(input_ids, task_ids=all_task_ids[i])
    dump_decision(model)


if __name__ == "__main__":
    main()

# %%
