# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import argparse
import itertools
import json
import os
import random
import time
import warnings
from functools import partial

# Recommend to initialize NUMA status at the most program begining (before any other imports)
from tutel_ea import system_init

system_init.init_affinity_at_program_beginning()


import brt.runtime.debug as brt_debug
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from brt.runtime.benchmark import deterministic_random_generator
from brt.runtime.pkg_info import BRT_CACHE_PATH
from brt.runtime.placement import dump_trace  # pylint: disable=unused-import
from brt.runtime.placement import (
    adaptive_load,
    adaptive_micro_bench_load,
    generate_experts_keys,
    generate_posible_placement,
)
from config import get_config
from data import build_loader
from logger import create_logger
from models import build_model
from models.micro_swin_v2_moe import MicroSwinV2TransformerMoE
from utils import create_ds_config, hook_scale_grad

warnings.filterwarnings(
    "ignore",
    "Argument interpolation should be of type InterpolationMode instead of int",
    UserWarning,
)

warnings.filterwarnings("ignore", "Leaking Caffe2 thread-pool after fork.", Warning)


def parse_option():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--custom_scaler", action="store_true", default=False)

    # deepspeed
    parser.add_argument("--enable_deepspeed", action="store_true", default=False)
    parser.add_argument(
        "--zero_opt", type=int, default=0, help="zero_optimization level"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="throughput",
        choices=[
            "throughput",
            "correctness",
            "trace",
            "gather-micro",
            "profile",
            "valid",
            "micro-bench",
        ],
    )
    parser.add_argument("--placement", type=str, default=None)
    parser.add_argument("--locality", action="store_true", default=False)
    ds_init = None

    args, _unparsed = parser.parse_known_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])

    config = get_config(args)
    config.defrost()
    config.MODEL.TYPE = "micro_swinv2_moe"
    config.freeze()

    return args, config, ds_init


def main(args, config, ds_init):
    (
        _dataset_train,
        _dataset_val,
        _data_loader_train,
        data_loader_val,
        _mixup_fn,
    ) = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    # logger.info(str(model))

    # For Tutel MoE
    for name, param in model.named_parameters():
        if hasattr(param, "skip_allreduce") and param.skip_allreduce is True:
            model.add_param_to_skip_allreduce(name)
            param.register_hook(partial(hook_scale_grad, config.TRAIN.MOE_GRAD_SCALE))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    flops = 0
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    model.cuda(config.LOCAL_RANK)
    model_without_ddp = model

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        bucket_cap_mb=64,
    )

    checkpoint_file = f"{config.MODEL.RESUME}.all_in_one"
    MOE_LAYER_VENDOR = os.environ.get("MOE_LAYER_VENDOR", "tutel")
    assert MOE_LAYER_VENDOR in ["brt_dist"]

    if args.mode == "gather-micro":
        adaptive_load(
            model_without_ddp,
            checkpoint_file,
            enable_locality=args.locality,
            global_expert_num=16,
        )
        torch.cuda.synchronize()
        dist.barrier()
        gather_micro_bench_data(data_loader_val, model, logger)
    elif args.mode == "micro-bench":
        micro_bench(config, model_without_ddp, 0, 1, checkpoint_file, logger)


@torch.inference_mode()
def micro_bench(
    config,
    model_without_ddp: MicroSwinV2TransformerMoE,
    moe_layer_start,
    moe_layer_end,
    checkpoint_file,
    logger,
):
    experts_range = {2: 18, 3: 2}
    experts_keys = generate_experts_keys(experts_range)
    start_moe_keys = experts_keys[moe_layer_start]
    end_moe_keys = experts_keys[moe_layer_end]
    assert start_moe_keys[0] == end_moe_keys[0]
    moe_blocks = model_without_ddp.layers[start_moe_keys[0]].blocks[
        start_moe_keys[1] : end_moe_keys[1]
    ]
    logger.info(moe_blocks)

    class MoEBlockWrapper(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.moe_blocks = blocks

        def forward(self, x):
            for moe_block in self.moe_blocks:
                x, _aux = moe_block(x)
            return x

    micro_moe_block = MoEBlockWrapper(moe_blocks)
    micro_moe_block_ddp = torch.nn.parallel.DistributedDataParallel(
        micro_moe_block,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        bucket_cap_mb=64,
    )
    micro_moe_block_ddp.eval()
    gpu_data = load_micro_bench_data(
        "swin_moe_micro_bench_data", config.DATA.BATCH_SIZE, logger
    )
    in_data = gpu_data[moe_layer_start]

    possible_placement = generate_posible_placement(16, dist.get_world_size())
    best_throughput = 0
    best_placement = None
    worst_throughput = 1e10
    worst_placement = None
    placement_num = len(possible_placement)
    for idx, placement in enumerate(possible_placement):
        adaptive_micro_bench_load(
            model_without_ddp, placement, end_moe_keys, checkpoint_file, 16
        )
        torch.cuda.synchronize()
        logger.info("===> Start throughput benchmark")
        dist.barrier()
        for _idx in range(20):
            micro_moe_block_ddp(in_data)
        torch.cuda.synchronize()
        dist.barrier()
        logger.info("Warmup done, start benchmarking")
        start = time.time()
        for _idx in range(200):
            micro_moe_block_ddp(in_data)
        torch.cuda.synchronize()
        dist.barrier()
        end = time.time()
        throughput = config.DATA.BATCH_SIZE * 200 / (end - start)
        if best_throughput < throughput:
            best_throughput = throughput
            best_placement = np.array(list(itertools.chain.from_iterable(placement)))
            if dist.get_rank() == 0:
                np.savetxt(
                    "best_placement.csv", best_placement, fmt="%s", delimiter=","
                )
        if worst_throughput > throughput:
            worst_throughput = throughput
            worst_placement = np.array(list(itertools.chain.from_iterable(placement)))
            if dist.get_rank() == 0:
                np.savetxt(
                    "worst_placement.csv", worst_placement, fmt="%s", delimiter=","
                )
        logger.info(
            f"{idx}/{placement_num} ===> Current Throughput: {throughput}, Best Throughput: {best_throughput}, Worst Throughput: {worst_throughput}, Gap: {best_throughput/worst_throughput:.2f}"
        )


def load_micro_bench_data(data_dir: str, bs: int, logger):
    logger.info(f"Loading micro-bench data from {data_dir}")
    data_dir_path = BRT_CACHE_PATH / f"datasets/{data_dir}"
    data_file_path = (
        data_dir_path
        / f"world_size_{dist.get_world_size()}"
        / f"rank{dist.get_rank()}_{bs}.npz"
    )
    data = np.load(data_file_path, allow_pickle=True)
    numpy_data = list(data.values())
    data_on_gpu = [torch.from_numpy(d).cuda() for d in numpy_data]
    return data_on_gpu


def get_benchmark_data(data_loader, logger, num_batches=100):
    max_batches = num_batches if len(data_loader) > num_batches else len(data_loader)
    logger.info(
        f"===> Preparing benchmark data, get first {max_batches} batches from data loader of length {len(data_loader)}"
    )
    gpu_data = []
    for idx, (images, _target) in enumerate(data_loader):
        if idx == max_batches:
            break
        gpu_data.append(images.cuda())
    batch_size = gpu_data[0].size(0)
    logger.info(f"===> Benchmark data prepared, batch size {batch_size}")
    return gpu_data, batch_size


@torch.inference_mode()
def gather_micro_bench_data(data_loader, model, logger):
    brt_debug.set_targeted_profile_position(target=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    bench_nums = 1
    data_iter = iter(data_loader)
    batch_size = 0
    for _idx in range(bench_nums):
        logger.info(f"===> Gathering micro benchmark data {(_idx + 1)}/{bench_nums}")
        images, _target = next(data_iter)
        batch_size = images.size(0)
        images = images.cuda(non_blocking=True)
        model(images)
    brt_debug.save_profile(
        profile_name=f"{batch_size}", profile_dir="datasets/swin_moe_micro_bench_data"
    )
    torch.cuda.synchronize()
    dist.barrier()
    logger.info(
        f"Debug Profile done, total data capatured: Batch size: {batch_size}, Num batches: {bench_nums}"
    )


@torch.inference_mode()
def throughput(data_loader, model, logger):
    bench_nums = 10
    gpu_data, batch_size = get_benchmark_data(data_loader, logger, bench_nums)
    bench_nums = len(gpu_data)
    torch.cuda.synchronize()
    logger.info("===> Start throughput benchmark")
    dist.barrier()
    for idx in range(20):
        model(gpu_data[idx % bench_nums])
    torch.cuda.synchronize()
    dist.barrier()
    logger.info("Warmup done, start benchmarking")
    start = time.time()
    for idx, data in enumerate(gpu_data):
        model(data)
    torch.cuda.synchronize()
    end = time.time()
    logger.info(
        f"Batch size: {batch_size}, Throughput: {len(gpu_data) * batch_size / (end - start)}"
    )


@torch.inference_mode()
def check_correctness(model, bs=1, iteration=10):
    model.eval()
    # timer = CUDATimer(1, 10, 5)
    inputs_generator = deterministic_random_generator(
        [bs, 3, 192, 192], num=iteration, dtype=torch.float32, device="cuda"
    )

    for inputs in inputs_generator:
        outputs = model(inputs)
        print(outputs[0])
        print(outputs[0].sum())
        input()


if __name__ == "__main__":
    args, config, ds_init = parse_option()

    # print("Environments:", os.environ)
    # print("Pytorch NCCL VERSION: ", torch.cuda.nccl.version())

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        master_port = os.environ["MASTER_PORT"]
        master_addr = os.environ["MASTER_ADDR"]
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(
            f"RANK and WORLD_SIZE in environ: {rank}/{world_size}, LOCAL_RANK: {config.LOCAL_RANK}, "
            f"master_node: {master_addr}:{master_port}"
        )
        if "OMPI_COMM_WORLD_RANK" in os.environ:
            del os.environ["OMPI_COMM_WORLD_RANK"]
        if "OMPI_COMM_WORLD_SIZE" in os.environ:
            del os.environ["OMPI_COMM_WORLD_SIZE"]
        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            del os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.distributed.barrier()

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = (
        config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    # gradient accumulation also need to scale the learning rate
    config.defrost()
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = (
            linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        )
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        if config.TRAIN.SAVE_STEP > 0:
            config.TRAIN.SAVE_STEP = (
                config.TRAIN.SAVE_STEP * config.TRAIN.ACCUMULATION_STEPS
            )
    config.TRAIN.BASE_LR = (
        config.TRAIN.TOTAL_LR if config.TRAIN.TOTAL_LR > 0 else linear_scaled_lr
    )
    config.TRAIN.WARMUP_LR = (
        config.TRAIN.TOTAL_WARMUP_LR
        if config.TRAIN.TOTAL_WARMUP_LR > 0
        else linear_scaled_warmup_lr
    )
    config.TRAIN.MIN_LR = (
        config.TRAIN.TOTAL_MIN_LR
        if config.TRAIN.TOTAL_MIN_LR > 0
        else linear_scaled_min_lr
    )
    config.freeze()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}"
    )

    if ds_init is not None:
        create_ds_config(args, config)

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        path = os.path.join(config.OUTPUT, "args.json")
        with open(path, "w") as f:
            f.write(json.dumps(vars(args)))
        logger.info(f"Full config saved to {path}")

    # print config
    # logger.info(config.dump())
    # logger.info(json.dumps(vars(args)))

    main(args, config, ds_init)
