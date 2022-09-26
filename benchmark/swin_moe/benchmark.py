# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

# Recommend to initialize NUMA status at the most program begining (before any other imports)
from tutel_ea import system_init

system_init.init_affinity_at_program_beginning()

import os
import time
import pickle
import json
import string
import random
import argparse
import datetime
import numpy as np
from functools import partial
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer, set_weight_decay
from logger import create_logger
from utils import (
    load_checkpoint,
    save_checkpoint,
    auto_resume_helper,
    reduce_tensor,
    create_ds_config,
    NativeScalerWithGradNormCount,
    load_pretrained,
    hook_scale_grad,
)

import warnings
from tutel_ea.moe import router_exporter
from brt.runtime.benchmark import CUDATimer, generate_deterministic_random_data

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
        "--single-gpu-eval",
        action="store_true",
        help="whether to do eval on single GPU",
    )
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    parser.add_argument("--custom_scaler", action="store_true", default=False)

    # distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        required=True,
        help="local rank for DistributedDataParallel",
    )

    # deepspeed
    parser.add_argument("--enable_deepspeed", action="store_true", default=False)
    parser.add_argument(
        "--zero_opt", type=int, default=0, help="zero_optimization level"
    )
    parser.add_argument(
        "--dpfp16", action="store_true", default=False, help="deepspeed fp16"
    )

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        raise NotImplementedError("Tutel MoE not support deepspeed now")
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig

            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.5'")
            ds_init = None
            exit(0)
    else:
        ds_init = None

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config, ds_init


def main(args, config, ds_init):
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    # For Tutel MoE
    for name, param in model.named_parameters():
        if hasattr(param, "skip_allreduce") and param.skip_allreduce is True:
            print(f"===>[rank{dist.get_rank()}] {name} skipping all_reduce")
            model.add_param_to_skip_allreduce(name)
            param.register_hook(partial(hook_scale_grad, config.TRAIN.MOE_GRAD_SCALE))
            print(
                f"[{name}] skip_allreduce and div {config.TRAIN.MOE_GRAD_SCALE} for grad"
            )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    flops = 0
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    model.cuda(config.LOCAL_RANK)
    model_without_ddp = model

    if config.ENABLE_DEEPSPEED:
        skip = {}
        skip_keywords = {}
        if hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
        if hasattr(model, "no_weight_decay_keywords"):
            skip_keywords = model.no_weight_decay_keywords()
        parameters = set_weight_decay(model, skip, skip_keywords)
        logger.info("========> DeepSpeed initializing......")
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=parameters,
            dist_init_required=False,
        )
        logger.info("========> DeepSpeed initializd!!!!!!")
        logger.info(
            f"model.gradient_accumulation_steps() = {model.gradient_accumulation_steps()}"
        )
        assert model.gradient_accumulation_steps() == config.TRAIN.ACCUMULATION_STEPS
        loss_scaler = None
    else:
        optimizer = build_optimizer(config, model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False,
            bucket_cap_mb=64,
        )
        loss_scaler = NativeScalerWithGradNormCount(args.custom_scaler)

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger, config.ENABLE_DEEPSPEED)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            logger,
            loss_scaler=loss_scaler,
        )
        debug(model, bs=1)
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
        )

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch + config.SEED)

        print(f"Epoch[{epoch}] starting....")
        train_one_epoch(
            config,
            model,
            model_without_ddp,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            loss_scaler,
        )
        print(f"Epoch[{epoch}] ending....")

        tic = time.time()
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                logger,
                loss_scaler=loss_scaler,
            )
        print(f"rank[{dist.get_rank()}] Save checkpoint takes {time.time() - tic}s")
        logger.info(
            f"rank[{dist.get_rank()}] Save checkpoint takes {time.time() - tic}s"
        )

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
        )
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    logger.info(f"number of params: {n_parameters}")
    logger.info(f"number of GFLOPs: {flops / 1e9}")


def train_one_epoch(
    config,
    model,
    model_without_ddp,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    loss_scaler,
):
    model.train()
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_aux_meter = AverageMeter()
    loss_cls_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    data_end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        if epoch == config.TRAIN.START_EPOCH:
            if idx < config.TRAIN.START_STEP:
                lr_scheduler.step_update(
                    (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
                )
                logger.info(f"passing Epoch[{epoch}] Step[{idx}]")
                continue
        data_time.update(time.time() - data_end)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            if config.DPFP16:
                samples = samples.half()
            outputs, l_aux = model(samples)
            loss_cls = criterion(outputs, targets)
            loss = loss_cls + l_aux * config.TRAIN.MOE_LOSS_WEIGHT
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            model.backward(loss)
            model.step()
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                # Deepspeed will call step() & model.zero_grad() automatic
                lr_scheduler.step_update(
                    (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
                )

            grad_norm = -1
            if config.DPFP16:
                loss_scale_value = (
                    model.optimizer.loss_scale
                    if hasattr(model.optimizer, "loss_scale")
                    else model.optimizer.cur_scale
                )
            else:
                loss_scale_value = -1
        else:
            with torch.cuda.amp.autocast():
                outputs, l_aux = model(samples)
                loss_cls = criterion(outputs, targets)
                loss = loss_cls + l_aux * config.TRAIN.MOE_LOSS_WEIGHT
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=config.TRAIN.CLIP_GRAD,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
            )
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update(
                    (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
                )
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(
            loss.item() * config.TRAIN.ACCUMULATION_STEPS, targets.size(0)
        )  # todo: modify in 2022/02/24
        loss_cls_meter.update(loss_cls.item(), targets.size(0))
        if isinstance(l_aux, float):
            loss_aux_meter.update(l_aux, targets.size(0))
        else:
            loss_aux_meter.update(l_aux.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()
        data_end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            wd = optimizer.param_groups[0]["weight_decay"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"data_time {data_time.val:.4f} ({data_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"loss-cls {loss_cls_meter.val:.4f} ({loss_cls_meter.avg:.4f})\t"
                f"loss-aux {loss_aux_meter.val:.4f} ({loss_aux_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )

        if config.TRAIN.SAVE_STEP > 0 and (idx + 1) % config.TRAIN.SAVE_STEP == 0:
            if (
                config.TRAIN.ACCUMULATION_STEPS > 1
                and (idx + 1) % config.TRAIN.ACCUMULATION_STEPS != 0
            ):
                pass
            else:
                tic = time.time()
                save_checkpoint(
                    config,
                    epoch,
                    model_without_ddp,
                    -1.0,
                    optimizer,
                    lr_scheduler,
                    logger,
                    step=idx,
                    loss_scaler=loss_scaler,
                )
                print(
                    f"rank[{dist.get_rank()}] Save checkpoint takes {time.time() - tic}s"
                )
                logger.info(
                    f"rank[{dist.get_rank()}] Save checkpoint takes {time.time() - tic}s"
                )
                batch_time.reset()
                data_time.reset()
                loss_meter.reset()
                loss_cls_meter.reset()
                loss_aux_meter.reset()
                norm_meter.reset()
                scaler_meter.reset()

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )


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
    expert_export_path = os.getenv("EXPORT_EXPERT_PATH")

    if expert_export_path:
        print(f"Enable expert export to path {expert_export_path}")
        router_exporter.set_path(expert_export_path)

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            # compute output
            if router_exporter.is_enabled():
                router_exporter.new_entry()
                # router_exporter.set_input(images)
            output, l_aux = model(images)
            if router_exporter.is_enabled():
                router_exporter.set_output(output)
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
    if expert_export_path:
        router_exporter.dump()
        exit()
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return


def debug(model, bs=1):
    model.eval()
    # timer = CUDATimer(1, 10, 5)
    timer = CUDATimer(0, 1, 1)
    inputs = generate_deterministic_random_data(
        [bs, 3, 192, 192], dtype=torch.float32, device="cuda"
    )
    timer.execute(lambda: model(inputs), "debugging")

    return


if __name__ == "__main__":
    args, config, ds_init = parse_option()

    print("Environments:", os.environ)
    print("Pytorch NCCL VERSION: ", torch.cuda.nccl.version())

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
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(args, config, ds_init)
