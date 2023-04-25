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
import pathlib
import random
import time
import warnings
from functools import partial
from typing import Dict, List, Tuple

# Recommend to initialize NUMA status at the most program begining (before any other imports)
from tutel_ea import system_init

system_init.init_affinity_at_program_beginning()


import brt.runtime.debug as brt_debug
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from brt.runtime.benchmark import ResultWriter, deterministic_random_generator
from brt.runtime.pkg_info import BRT_CACHE_PATH

# pylint: disable=unused-import
from brt.runtime.placement import (
    adaptive_load,
    adaptive_micro_bench_load,
    deterministic_rand_placement_generator,
    dump_trace,
    generate_experts_keys,
    permute_placement,
    possible_placement_generator,
)
from config import get_config
from data import build_loader
from logger import create_logger
from models import build_model
from models.micro_swin_v2_moe import MicroSwinV2TransformerMoE
from utils import create_ds_config, hook_scale_grad

# pylint: enable=unused-import


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
        choices=["gather-micro", "search-end", "bench-searched", "bench-permuted"],
    )
    parser.add_argument("--placement", type=str, default=None)
    parser.add_argument(
        "--capacity",
        type=float,
        default=1.25,
        help="capacity",
        choices=[1.25, 2.0, 3.0, 4.0],
    )
    parser.add_argument(
        "--moe-id", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    parser.add_argument("--seed", type=int, default=0)

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
            enable_locality=False,
            global_expert_num=16,
        )
        torch.cuda.synchronize()
        dist.barrier()
        gather_micro_bench_data(config, data_loader_val, model, logger)
    elif args.mode == "search-end":
        search_end_layer_placement(
            config,
            model_without_ddp,
            args.moe_id,
            args.moe_id,
            args.seed,
            checkpoint_file,
            logger,
        )
    elif args.mode == "bench-searched":
        benchmark_serached_placement(
            config,
            model_without_ddp,
            args.moe_id,
            args.moe_id,
            checkpoint_file,
            logger,
        )
    elif args.mode == "bench-permuted":
        benchmark_permuted_serached_placement(
            config, model_without_ddp, 0, 1, checkpoint_file, logger
        )


def make_micro_bench_model(
    config,
    model_without_ddp: MicroSwinV2TransformerMoE,
    moe_layer_start: int,
    moe_layer_end: int,
    logger,
):
    experts_range = {2: 18, 3: 2}
    experts_keys = generate_experts_keys(experts_range)
    start_moe_keys = experts_keys[moe_layer_start]
    end_moe_keys = experts_keys[moe_layer_end]
    assert start_moe_keys[0] == end_moe_keys[0]
    moe_blocks = model_without_ddp.layers[start_moe_keys[0]].blocks[
        start_moe_keys[1] : end_moe_keys[1] + 1
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
        config, "swin_moe_micro_bench_data", config.DATA.BATCH_SIZE, logger
    )

    in_data = gpu_data[moe_layer_start]

    return start_moe_keys, end_moe_keys, micro_moe_block_ddp, in_data


@torch.inference_mode()
def benchmark_serached_placement(
    config,
    model_without_ddp: MicroSwinV2TransformerMoE,
    moe_layer_start: int,
    moe_layer_end: int,
    checkpoint_file: str,
    logger,
):
    _, _, micro_moe_block_ddp, in_data = make_micro_bench_model(
        config, model_without_ddp, moe_layer_start, moe_layer_end, logger
    )
    best_searched_placement = load_searched_placement(
        config, "best", moe_layer_start, moe_layer_end, logger
    )
    worst_searched_placement = load_searched_placement(
        config, "worst", moe_layer_start, moe_layer_end, logger
    )
    adaptive_micro_bench_load(
        model_without_ddp, best_searched_placement, checkpoint_file, 16
    )
    best_througput = benchmark_micro_ddp_model(
        config, micro_moe_block_ddp, in_data, "best", logger
    )
    adaptive_micro_bench_load(
        model_without_ddp, worst_searched_placement, checkpoint_file, 16
    )
    worst_througput = benchmark_micro_ddp_model(
        config, micro_moe_block_ddp, in_data, "worst", logger
    )
    result_fname = f"swin_moe/micro_throughput.csv"
    result_writer = ResultWriter(result_fname)
    result_writer.write(
        f"{config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR},{moe_layer_start+1},{best_througput/worst_througput}"
    )
    logger.info(
        f"moe blocks: {moe_layer_start} - {moe_layer_end}, speedup: {best_througput/worst_througput}"
    )


@torch.inference_mode()
def benchmark_permuted_serached_placement(
    config,
    model_without_ddp: MicroSwinV2TransformerMoE,
    moe_layer_start: int,
    moe_layer_end: int,
    checkpoint_file: str,
    logger,
):
    _, end_moe_key, micro_moe_block_ddp, in_data = make_micro_bench_model(
        config, model_without_ddp, moe_layer_start, moe_layer_end, logger
    )
    best_searched_placement = load_searched_placement(
        config, "best", moe_layer_start, moe_layer_end, logger
    )
    worst_searched_placement = load_searched_placement(
        config, "worst", moe_layer_start, moe_layer_end, logger
    )
    adaptive_micro_bench_load(
        model_without_ddp, best_searched_placement, checkpoint_file, 16
    )
    best_througput = benchmark_micro_ddp_model(
        config, micro_moe_block_ddp, in_data, "best", logger
    )
    adaptive_micro_bench_load(
        model_without_ddp, worst_searched_placement, checkpoint_file, 16
    )
    worst_througput = benchmark_micro_ddp_model(
        config, micro_moe_block_ddp, in_data, "worst", logger
    )
    logger.info(
        f"Original: moe blocks: {moe_layer_start} - {moe_layer_end}, speedup: {best_througput/worst_througput}"
    )

    all_best_througput = 0
    all_worst_througput = 0
    for i in range(10):
        placement = best_searched_placement[end_moe_key]
        best_searched_placement[end_moe_key] = permute_placement(placement, i)
        # print(f"best placement: {best_searched_placement}")
        placement = worst_searched_placement[end_moe_key]
        worst_searched_placement[end_moe_key] = permute_placement(placement, i)
        # print(f"worst placement: {worst_searched_placement}")
        adaptive_micro_bench_load(
            model_without_ddp, best_searched_placement, checkpoint_file, 16
        )
        best_througput = benchmark_micro_ddp_model(
            config, micro_moe_block_ddp, in_data, "best", logger
        )
        all_best_througput += best_througput
        adaptive_micro_bench_load(
            model_without_ddp, worst_searched_placement, checkpoint_file, 16
        )
        worst_througput = benchmark_micro_ddp_model(
            config, micro_moe_block_ddp, in_data, "worst", logger
        )
        all_worst_througput += worst_througput
        logger.info(
            f"Permutation iter: {i}, moe blocks: {moe_layer_start} - {moe_layer_end}, speedup: {best_througput/worst_througput}"
        )
    logger.info(f"Average permuted speedup: {all_best_througput/all_worst_througput}")


@torch.inference_mode()
def benchmark_micro_ddp_model(config, model, in_data, item, logger):
    model.eval()
    torch.cuda.synchronize()
    logger.info(f"===> Start {item} throughput benchmark")
    dist.barrier()
    # print(micro_moe_block_ddp(in_data).sum())
    for _idx in range(20):
        # print(in_data.sum())
        model(in_data)
    torch.cuda.synchronize()
    dist.barrier()
    # logger.info("Warmup done, start benchmarking")
    start = time.time()
    for _idx in range(40):
        model(in_data)
    torch.cuda.synchronize()
    dist.barrier()
    end = time.time()
    throughput = config.DATA.BATCH_SIZE * 40 / (end - start)
    return throughput


@torch.inference_mode()
def search_end_layer_placement(
    config,
    model_without_ddp: MicroSwinV2TransformerMoE,
    moe_layer_start: int,
    moe_layer_end: int,
    seed: int,
    checkpoint_file: str,
    logger,
):
    _, end_moe_keys, micro_moe_block_ddp, in_data = make_micro_bench_model(
        config, model_without_ddp, moe_layer_start, moe_layer_end, logger
    )
    micro_moe_block_ddp.eval()

    placement_generator = deterministic_rand_placement_generator(
        16, dist.get_world_size(), seed
    )
    best_throughput = 0
    best_placement = None
    worst_throughput = 1e10
    worst_placement = None
    idx = 0
    micro_results_path = BRT_CACHE_PATH / "results" / "swin_moe" / "micro_results"
    micro_results_path.mkdir(parents=True, exist_ok=True)
    no_update_iter_nums = 0
    for i in range(1000):
        placement = next(placement_generator)
        idx += 1
        modified_placements = {}
        modified_placements[end_moe_keys] = placement
        adaptive_micro_bench_load(
            model_without_ddp, modified_placements, checkpoint_file, 16
        )
        torch.cuda.synchronize()
        logger.info("===> Start throughput benchmark")
        dist.barrier()
        # print(micro_moe_block_ddp(in_data).sum())
        for _idx in range(20):
            # print(in_data.sum())
            micro_moe_block_ddp(in_data)
        torch.cuda.synchronize()
        dist.barrier()
        logger.info("Warmup done, start benchmarking")
        start = time.time()
        for _idx in range(40):
            micro_moe_block_ddp(in_data)
        torch.cuda.synchronize()
        dist.barrier()
        end = time.time()
        throughput = config.DATA.BATCH_SIZE * 40 / (end - start)
        if best_throughput < throughput:
            best_throughput = throughput
            best_placement = np.array(list(itertools.chain.from_iterable(placement)))
            if dist.get_rank() == 0:
                np.savetxt(
                    micro_results_path
                    / f"{moe_layer_start}_{moe_layer_end}.{config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR}.best_{dist.get_world_size()}_placement.csv",
                    best_placement,
                    fmt="%s",
                    delimiter=",",
                )
            no_update_iter_nums = 0

        if worst_throughput > throughput:
            worst_throughput = throughput
            worst_placement = np.array(list(itertools.chain.from_iterable(placement)))
            if dist.get_rank() == 0:
                np.savetxt(
                    micro_results_path
                    / f"{moe_layer_start}_{moe_layer_end}.{config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR}.worst_{dist.get_world_size()}_placement.csv",
                    worst_placement,
                    fmt="%s",
                    delimiter=",",
                )
            no_update_iter_nums = 0
        no_update_iter_nums += 1
        gap = best_throughput / worst_throughput
        logger.info(
            f"{idx} ===>Current Throughput: {throughput}, Best Throughput: {best_throughput}, Worst Throughput: {worst_throughput}, Gap: {gap:.2f}"  # pylint: disable=line-too-long
        )
        if no_update_iter_nums > 500 and gap <= 1.05:
            break
    if dist.get_rank() == 0:
        result_path = pathlib.Path(
            micro_results_path / f"world_size_{dist.get_world_size()}.csv"
        )
        result = result_path.open("a")
        result.write(
            f"{moe_layer_start}_{moe_layer_end},{config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR},{best_throughput:.2f},{worst_throughput:.2f},{best_throughput/worst_throughput:.2f}\n",
        )


def load_searched_placement(
    config, which_one: str, moe_layer_start: int, moe_layer_end: int, logger
) -> Dict[Tuple[int, int], List[List[int]]]:
    result_path = BRT_CACHE_PATH / "results" / "swin_moe"
    world_size = dist.get_world_size()
    experts_range = {2: 18, 3: 2}
    experts_keys = generate_experts_keys(experts_range)
    capacity_factor = config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR
    assert which_one in ["best", "worst"]
    searched_placement_file = (
        result_path
        / f"micro_results/{moe_layer_start}_{moe_layer_end}.{capacity_factor}.{which_one}_{world_size}_placement.csv"
    )
    logger.info(f"Loading searched placement from {searched_placement_file.as_posix()}")

    searched_placement_list = np.loadtxt(
        searched_placement_file, dtype=np.int32, delimiter=","
    )
    searched_placement_list = searched_placement_list.reshape(-1, 16)
    assert len(searched_placement_list) == moe_layer_end - moe_layer_start + 1
    searched_placement = {}
    for i in range(moe_layer_start, moe_layer_end + 1):
        placement = np.split(searched_placement_list[i - moe_layer_start], world_size)
        placement = [list(p) for p in placement]
        searched_placement[experts_keys[i]] = placement
    return searched_placement


def load_micro_bench_data(config, data_dir: str, bs: int, logger):
    logger.info(f"Loading micro-bench data from {data_dir}")
    data_dir_path = BRT_CACHE_PATH / f"dataset/{data_dir}"
    data_file_path = (
        data_dir_path
        / f"world_size_{dist.get_world_size()}"
        / f"rank{dist.get_rank()}_{bs}_{config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR}.npz"
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
def gather_micro_bench_data(config, data_loader, model, logger):
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
        profile_name=f"{batch_size}_{config.MODEL.SWIN_V2_MOE.CAPACITY_FACTOR}",
        profile_dir="dataset/swin_moe_micro_bench_data",
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
        output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name="micro"
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
