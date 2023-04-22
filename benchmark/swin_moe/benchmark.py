# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import argparse
import json
import os
import pathlib
import random
import time
import warnings
from functools import partial

# Recommend to initialize NUMA status at the most program begining (before any other imports)
from tutel_ea import system_init

system_init.init_affinity_at_program_beginning()


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from brt.runtime import BRT_CACHE_PATH
from brt.runtime.benchmark import deterministic_random_generator, profile_v2
from brt.runtime.placement import (
    adaptive_load,
    adaptive_micro_bench_load,
    dump_trace,
    load_searched_placement,
)
from config import get_config
from data import build_loader
from logger import create_logger
from models import build_model
from timm.utils import AverageMeter, accuracy
from utils import (
    adaptive_load_checkpoint,
    create_ds_config,
    gather_all_ckpts_into_one,
    hook_scale_grad,
    reduce_tensor,
)

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
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only", default=False
    )
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
            "gather-ckpt",
            "profile",
            "valid",
        ],
    )
    parser.add_argument("--placement", action="store_true", default=False)
    parser.add_argument("--capacity", type=float, default=1.25)
    ds_init = None

    args, _unparsed = parser.parse_known_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])

    config = get_config(args)

    return args, config, ds_init


def main(args, config, ds_init):
    (
        _dataset_train,
        dataset_val,
        _data_loader_train,
        data_loader_val,
        _mixup_fn,
    ) = build_loader(config)

    # logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    # logger.info(str(model))

    # For Tutel MoE
    for name, param in model.named_parameters():
        if hasattr(param, "skip_allreduce") and param.skip_allreduce is True:
            model.add_param_to_skip_allreduce(name)
            param.register_hook(partial(hook_scale_grad, config.TRAIN.MOE_GRAD_SCALE))

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"number of params: {n_parameters}")
    # flops = 0
    # # if hasattr(model, "flops"):
    #     flops = model.flops()
    # logger.info(f"number of GFLOPs: {flops / 1e9}")
    model.cuda(config.LOCAL_RANK)
    model_without_ddp = model

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        bucket_cap_mb=64,
    )

    if args.mode == "gather-ckpt":
        gather_all_ckpts_into_one(config, model_without_ddp, logger)
    else:
        checkpoint_file = f"{config.MODEL.RESUME}.all_in_one"
        MOE_LAYER_VENDOR = os.environ.get("MOE_LAYER_VENDOR", "tutel")
        if MOE_LAYER_VENDOR == "brt_dist" and args.placement:
            logger.info("===> Loading searched placement")
            searched_placedment = load_searched_placement(config, "best")
            adaptive_micro_bench_load(
                model_without_ddp, searched_placedment, checkpoint_file, 16
            )
        else:
            adaptive_load(
                model_without_ddp,
                checkpoint_file,
                False,
                global_expert_num=16,
            )
        torch.cuda.synchronize()
        dist.barrier()
        if args.mode == "correctness":
            print("===> Running correctness test")
            check_correctness(model, args.batch_size)
        elif args.mode == "trace":
            print("===> Tracing model")
            adaptive_load_checkpoint(
                config,
                model_without_ddp,
                logger,
            )
            acc1, _acc5, _loss = validate(config, data_loader_val, model)
            logger.info(
                f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
            )
            print("===> Tracing model done, dumping to file")
            dump_trace(model_without_ddp)

        elif args.mode == "throughput":
            throughput(data_loader_val, model, logger)
        elif args.mode == "profile":
            gpu_data, _batch_size = get_benchmark_data(data_loader_val, logger, 20)
            profile_v2(model, gpu_data, MOE_LAYER_VENDOR)
        elif args.mode == "valid":
            validate(config, data_loader_val, model)


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_aux_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            # compute output
            output, l_aux = model(images)
            # measure accuracy and record loss
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        if isinstance(l_aux, float):
            loss_aux_meter.update(l_aux, target.size(0))
        else:
            loss_aux_meter.update(l_aux.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss-Cls {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Loss-Aux {loss_aux_meter.val:.4f} ({loss_aux_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    logger.info(f" * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}")
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def get_benchmark_data(data_loader, logger, num_batches=100):
    max_batches = num_batches if len(data_loader) > num_batches else len(data_loader)
    # logger.info(
    #     f"===> Preparing benchmark data, get first {max_batches} batches from data loader of length {len(data_loader)}"
    # )

    gpu_data = []
    for idx, (images, _target) in enumerate(data_loader):
        if idx == max_batches:
            break
        gpu_data.append(images.cuda())
    batch_size = gpu_data[0].size(0)
    # logger.info(f"===> Benchmark data prepared, batch size {batch_size}")
    return gpu_data, batch_size


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

    if dist.get_rank() == 0:
        result_path = BRT_CACHE_PATH / "results" / "swin_moe" / "e2e.csv"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_f = result_path.open("a")
        # result_f.write()
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
        # print(
        #     f"RANK and WORLD_SIZE in environ: {rank}/{world_size}, LOCAL_RANK: {config.LOCAL_RANK}, "
        #     f"master_node: {master_addr}:{master_port}"
        # )
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
        # logger.info(f"Full config saved to {path}")

        path = os.path.join(config.OUTPUT, "args.json")
        with open(path, "w") as f:
            f.write(json.dumps(vars(args)))
        # logger.info(f"Full config saved to {path}")

    # print config
    # logger.info(config.dump())
    # logger.info(json.dumps(vars(args)))

    main(args, config, ds_init)
